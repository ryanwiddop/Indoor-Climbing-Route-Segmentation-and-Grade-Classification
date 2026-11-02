# pair:  PR-AUC = 91.39%  Precision = 82.99%  Recall = 87.11%  F1 = 85.00%  @0.40
# route: recovery@0.8 IoU = 78.43% macro / 72.81% micro  @0.30, average linkage

from collections import defaultdict
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix

NUM_BOOST_ROUND = 400
N_JOBS = 12
IOU_MATCH = 0.8
OUT_DIR = "models/figures/phase_2"
CATEGORICAL = ["cat_min", "cat_max"]
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0,
    "nthread": 4,
}


def fit_fold(X, y, held_out):
    dtrain = xgb.DMatrix(X[~held_out], label=y[~held_out], enable_categorical=True)
    model = xgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    dtest = xgb.DMatrix(X[held_out], enable_categorical=True)
    return model.predict(dtest), model.get_score(importance_type="gain")


def group_wall(pairs, threshold, linkage="average"):
    ids = sorted(set(pairs["i_id"]) | set(pairs["j_id"]))
    index_of = {hold_id: k for k, hold_id in enumerate(ids)}
    n = len(ids)

    dist = np.ones((n, n))
    np.fill_diagonal(dist, 0.0)
    rows = pairs["i_id"].map(index_of).to_numpy()
    cols = pairs["j_id"].map(index_of).to_numpy()
    d = 1.0 - pairs["oof_probs"].to_numpy()
    dist[rows, cols] = d
    dist[cols, rows] = d

    if n == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = AgglomerativeClustering(
            metric="precomputed",
            linkage=linkage,
            n_clusters=None,
            distance_threshold=1.0 - threshold,
        ).fit_predict(dist)

    clusters = defaultdict(set)
    for hold_id, label in zip(ids, labels):
        clusters[int(label)].add(int(hold_id))
    return [clusters[label] for label in sorted(clusters)]


def image_gt_routes(pairs):
    routes = defaultdict(set)
    for col_id, col_route in (("i_id", "i_route"), ("j_id", "j_route")):
        for hold_id, route in zip(pairs[col_id], pairs[col_route]):
            routes[route].add(int(hold_id))
    return {str(r): s for r, s in sorted(routes.items())}


def route_iou(gt_route, pred_route):
    union = gt_route | pred_route
    return len(gt_route & pred_route) / len(union) if union else 0.0


def match_routes(gt_routes, pred_routes):
    if not gt_routes or not pred_routes:
        return []

    gt_keys = list(gt_routes)
    iou_matrix = np.zeros((len(gt_keys), len(pred_routes)))
    for i, gt_key in enumerate(gt_keys):
        for j, pred_route in enumerate(pred_routes):
            iou_matrix[i, j] = route_iou(gt_routes[gt_key], pred_route)

    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    return [
        {
            "gt_route": gt_keys[gt_ind],
            "pred_ind": int(pred_ind),
            "iou": float(iou_matrix[gt_ind, pred_ind]),
        }
        for gt_ind, pred_ind in zip(gt_indices, pred_indices)
    ]


def wall_recovery(pairs, threshold, linkage="average"):
    gt_routes = image_gt_routes(pairs)
    pred_routes = group_wall(pairs, threshold, linkage)
    matches = match_routes(gt_routes, pred_routes)
    hit = sum(1 for m in matches if m["iou"] >= IOU_MATCH)
    return hit / len(gt_routes), gt_routes, pred_routes, matches


def macro_recovery(all_pairs, threshold, linkage="average"):
    return np.array([
        wall_recovery(pairs, threshold, linkage)[0]
        for _, pairs in all_pairs.groupby("image_file")
    ])


def main():
    all_pairs = pd.read_csv("all_pairs.csv")
    for col in CATEGORICAL:
        all_pairs[col] = all_pairs[col].astype("category")

    walls = sorted(all_pairs["image_file"].unique())
    wall_of = all_pairs["image_file"].values
    X = all_pairs.drop(columns=["same_route", "image_file", "i_id", "j_id", "i_route", "j_route"])
    y = all_pairs["same_route"].astype(int)

    folds = Parallel(n_jobs=N_JOBS)(
        delayed(fit_fold)(X, y, wall_of == wall) for wall in walls
    )

    oof = np.zeros(len(all_pairs))
    fold_ap = []
    for wall, (probs, _) in zip(walls, folds):
        held_out = wall_of == wall
        oof[held_out] = probs
        fold_ap.append(average_precision_score(y[held_out], probs))
    fold_ap = np.array(fold_ap)

    print(f"{len(walls)} walls, {len(all_pairs)} pairs, {y.mean():.2%} positive")
    print(f"per-wall PR-AUC: mean {fold_ap.mean():.3f}  sd {fold_ap.std():.3f}  "f"min {fold_ap.min():.3f}  max {fold_ap.max():.3f}")
    print(f"pooled PR-AUC:   {average_precision_score(y, oof):.3f}\n")
    for wall, ap in sorted(zip(walls, fold_ap), key=lambda t: t[1]):
        print(f"  {wall}  {ap:.3f}")

    print()
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.30, 0.95, 0.05):
        preds = (oof >= threshold).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"{threshold:.2f}  {precision_score(y, preds, zero_division=0):.4f}  "f"{recall_score(y, preds, zero_division=0):.4f}  {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    preds = (oof >= best_threshold).astype(int)
    print(f"\nbest threshold {best_threshold:.2f}")
    print("PR-AUC:", average_precision_score(y, oof))
    print("Precision:", precision_score(y, preds, zero_division=0))
    print("Recall:", recall_score(y, preds, zero_division=0))
    print("F1:", f1_score(y, preds, zero_division=0))
    print(confusion_matrix(y, preds))

    importance = {}
    for _, gains in folds:
        for feature, gain in gains.items():
            importance[feature] = importance.get(feature, 0.0) + gain / len(folds)
    total = sum(importance.values())
    print()
    for feature, score in sorted(importance.items(), key=lambda t: -t[1]):
        print(f"{score / total:.2%}: {feature}")

    # np.save("oof_probs.npy", oof)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_pairs["oof_probs"] = oof
    all_pairs["preds"] = (oof >= best_threshold).astype(int)
    all_pairs.to_csv(f"{OUT_DIR}/all_pairs_with_preds.csv", index=False)

    print()
    sweep = []
    best_route_threshold, best_recovery = 0.5, -1.0
    for threshold in np.arange(0.20, 0.95, 0.05):
        per_wall = macro_recovery(all_pairs, threshold)
        preds = (oof >= threshold).astype(int)
        sweep.append({
            "threshold": float(threshold),
            "recovery_mean": float(per_wall.mean()),
            "recovery_sd": float(per_wall.std()),
            "pair_precision": float(precision_score(y, preds, zero_division=0)),
            "pair_recall": float(recall_score(y, preds, zero_division=0)),
            "pair_f1": float(f1_score(y, preds, zero_division=0)),
        })
        print(f"{threshold:.2f}  recovery@{IOU_MATCH} {per_wall.mean():.4f}  sd {per_wall.std():.4f}")
        if per_wall.mean() > best_recovery:
            best_recovery, best_route_threshold = per_wall.mean(), threshold

    predicted_routes = {}
    gt_routes = {}
    all_matches = {}
    per_wall = {}
    for image_file, pairs in all_pairs.groupby("image_file"):
        recovery, gt, pred, matches = wall_recovery(pairs, best_route_threshold)
        per_wall[image_file] = recovery
        gt_routes[image_file] = {r: sorted(s) for r, s in gt.items()}
        predicted_routes[image_file] = {str(k): sorted(s) for k, s in enumerate(pred)}
        all_matches[image_file] = matches

    recoveries = np.array(list(per_wall.values()))
    print(f"\nroute threshold {best_route_threshold:.2f}")
    print(f"recovery@{IOU_MATCH}: macro {recoveries.mean():.4f}  sd {recoveries.std():.4f}  "
          f"min {recoveries.min():.4f}  max {recoveries.max():.4f}")
    n_gt = sum(len(r) for r in gt_routes.values())
    n_hit = sum(1 for ms in all_matches.values() for m in ms if m["iou"] >= IOU_MATCH)
    n_pred = sum(len(r) for r in predicted_routes.values())
    print(f"micro: {n_hit}/{n_gt} = {n_hit / n_gt:.4f}   predicted routes {n_pred}")
    ious = np.array([m["iou"] for ms in all_matches.values() for m in ms])
    print(f"matched IoU: median {np.median(ious):.3f}  >=0.5 {(ious >= 0.5).mean():.3f}")
    print()
    for image_file, recovery in sorted(per_wall.items(), key=lambda t: t[1]):
        print(f"  {image_file}  {recovery:.3f}  "
              f"gt {len(gt_routes[image_file])}  pred {len(predicted_routes[image_file])}")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({
            "iou_match": IOU_MATCH,
            "pair_threshold": float(best_threshold),
            "route_threshold": float(best_route_threshold),
            "fold_ap": {w: float(a) for w, a in zip(walls, fold_ap)},
            "importance": {k: float(v) for k, v in importance.items()},
            "sweep": sweep,
            "per_wall_recovery": {w: float(r) for w, r in per_wall.items()},
        }, f, indent=1)

    with open(f"{OUT_DIR}/predicted_routes.json", "w") as f:
        json.dump({"threshold": float(best_route_threshold),
                   "predicted": predicted_routes,
                   "gt": gt_routes,
                   "matches": all_matches}, f, indent=1)


if __name__ == "__main__":
    main()
