"""
Phase 3: Predict the route's V grade

Trains on annotated data (annotations.json).
Grades Phase 2 predictions (predicted_routes.json).
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
# from matplotlib.path import Path >> used for volumes, not implemented 11/16
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "data_preprocessing"))
from prepare_tabular import IGNORE_CATEGORIES, image_scale, load_coco

ANNOTATIONS_PATH = "/home/public/rwiddop/annotations.json"
PREDICTED_ROUTES = "models/figures/phase_2/predicted_routes.json"
OUT_DIR = "models/figures/phase_3"
MIN_HOLDS = 3
IOU_MATCH = 0.8

GRADES = ["V0-", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8+"]
GRADE_INDEX = {grade: i for i, grade in enumerate(GRADES)}

def to_grade(raw):
    """Convert raw grade prediction to the nearest valid grade index. (Rounding)"""
    return np.clip(np.rint(raw), 0, len(GRADES) - 1).astype(int)

def load_routes():
    coco = load_coco(ANNOTATIONS_PATH)
    ann_by_id = {ann["id"]: ann for ann in coco.raw["annotations"]}
    gt = {} # wall -> route_id -> [ann, ...]
    
    for ann in coco.raw["annotations"]:
        route_id = ann.get("attributes", {}).get("route_id")
        if route_id is None or coco.cat_id_to_name[ann["category_id"]] in IGNORE_CATEGORIES:
            continue
        wall = coco.img_by_id[ann["image_id"]]["file_name"]
        gt.setdefault(wall, {}).setdefault(route_id, []).append(ann)
        
    with open(PREDICTED_ROUTES) as f:
        predicted = json.load(f)["predicted"] # wall -> cluster -> [hold id, ...]
        
    pred = {
        wall: {cluster: [ann_by_id[h] for h in holds_ids] for cluster, holds_ids in clusters.items()}
        for wall, clusters in predicted.items()
    }
    
    return coco, gt, pred

HOLD_TYPES = ["jug", "crimp", "pocket", "sloper", "pinch", "edge", "foot"]
HAND_TYPES = set(HOLD_TYPES) - {"foot"}

def centre(ann):
    x, y, w, h = ann["bbox"]
    return np.array([x + w / 2, y + h / 2])

def stats(prefix, values):
    # mean / max / min / std of a list, NaN if the list is empty
    if len(values) == 0:
        return{f"{prefix}_{s}": np.nan for s in ("mean", "max", "min", "std")}
    values = np.asarray(values, dtype = float)
    
    return {
        f"{prefix}_mean" : values.mean(),
        f"{prefix}_max" : values.max(),
        f"{prefix}_min" : values.min(),
        f"{prefix}_std" : values.std()
    }
    
def nearest(points_from, points_to, same_set = False):
    # Distance from each point in points_from to nearest point in points_to
    if len(points_from) == 0 or len(points_to) - int(same_set) <=0:
        return []
    
    d = np.linalg.norm(points_from[:, None, :] - points_to[None, :, :], axis = 2)
    
    if same_set:
        np.fill_diagonal(d, np.inf)
        
    return d.min(axis = 1)

def route_features(anns, type_of, scale):
    # Features from hold types and distances
    types = [type_of(a) for a in anns]
    hands = np.array([centre(a) for a, t in zip(anns, types) if t in HAND_TYPES]).reshape(-1, 2) / scale
    feet = np.array([centre(a) for a, t in zip(anns, types) if t == "foot"]).reshape(-1, 2) / scale
    
    # hold types
    counts = Counter(types)
    row = {"n_holds": len(anns), "n_hands": len(hands), "n_feet": len(feet)}
    
    for t in HOLD_TYPES:
        row[f"n_{t}"] = counts[t]
        row[f"frac_{t}"] = counts[t] / len(anns)
        
    # moves: orders hand holds for bottom to top
    climb = hands[np.argsort(-hands[:, 1])] if len(hands) else hands
    moves = np.linalg.norm(np.diff(climb, axis = 0), axis = 1) if len(climb) > 1 else []
    
    row.update(stats("move", moves))
    row.update(stats("hand_nn", nearest(hands, hands, same_set = True)))
    row.update(stats("hand_to_foot", nearest(hands, feet)))
    row.update(stats("foot_nn", nearest(feet, feet, same_set = True)))
    
    #overall size of the route
    all_pts = np.vstack([hands, feet])
    row["height"] = np.ptp(all_pts[:, 1])
    row["width"] = np.ptp(all_pts[:, 0])
    
    return row

def build_table(coco, routes_by_wall):
    file_to_id = {img["file_name"]: img["id"] for img in coco.raw["images"]}
    type_of = lambda ann: coco.cat_id_to_name[ann["category_id"]]
    rows = []
    
    for wall, routes in sorted (routes_by_wall.items()):
        image_id = file_to_id[wall]
        scale = image_scale(coco.img_id_to_anns[image_id], coco)
        
        for route, anns in routes.items():
            anns = [a for a in anns if type_of(a) in HOLD_TYPES]
            
            if len(anns) < MIN_HOLDS:
                continue
            
            rows.append({"wall": wall, "route": str(route), "hold_ids": sorted(a["id"] for a in anns), **route_features(anns, type_of, scale)})
            
    return pd.DataFrame(rows)

def build_owner_lookup(gt):
    owner_of, holds_of, grade_of = {}, {}, {}
    
    for wall, routes in gt.items():
        for route_id, anns in routes.items():
            key = (wall, route_id)
            holds_of[key] = {a["id"] for a in anns}
            grade_of[key] = anns[0]["attributes"].get("route_grade", "INC")
            
            for a in anns:
                owner_of[a["id"]] = key
                
    return owner_of, holds_of, grade_of

def feature_columns(table):
    return sorted([c for c in table.columns if c not in ("wall", "route", "hold_ids", "y", "iou")])

def add_labels(table, owner_of, holds_of, grade_of):
    # y is the grade of the human-labeled route. iou = how well they match
    ys, ious = [], []
    
    for hold_ids in table["hold_ids"]:
        owners = Counter(owner_of[h] for h in hold_ids if h in owner_of)
        
        if not owners:
            ys.append(np.nan)
            ious.append(0.0)
            continue
        
        key = owners.most_common(1)[0][0]
        mine, theirs = set(hold_ids), holds_of[key]
        ious.append(len(mine & theirs) / len(mine | theirs))
        ys.append(GRADE_INDEX.get(grade_of[key], np.nan)) # INC -> NaN
        
    table["y"] = ys
    table["iou"] = ious

def score(y_true, raw):
    y_true = np.asarray(y_true, dtype = int)
    y_pred = to_grade(raw)
    
    return {
        "n": len(y_true),
        "exact": float(np.mean(y_pred == y_true)),
        "within_1": float(np.mean(np.abs(y_pred - y_true) <= 1)),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels = list(range(len(GRADES)))))
    }
    
def leave_one_wall_out(make_model, train, features, extra):
    # Train and extra wall's routes are predicted by a model trained on all other walls
    train_raw = pd.Series(np.nan, index = train.index)
    extra_raw = pd.Series(np.nan, index = extra.index)
    
    for wall in sorted(train["wall"].unique()):
        held = train["wall"] == wall
        model = make_model()
        model.fit(train.loc[~held, features], train.loc[~held, "y"])
        train_raw[held] = model.predict(train.loc[held, features])
        on_wall = extra["wall"] == wall
        
        if on_wall.any():
            extra_raw[on_wall] = model.predict(extra.loc[on_wall, features])
    
    return train_raw, extra_raw

MODELS = {
    "baseline (median)": lambda: DummyRegressor(strategy = "median"),
    "ridge": lambda: make_pipeline(SimpleImputer(strategy = "median"), StandardScaler(), Ridge(alpha = 10.0)),
    "xgboost": lambda: xgb.XGBRegressor(
        n_estimators = 300, max_depth = 3, learning_rate = 0.03, subsample = 0.8,
        colsample_bytree = 0.6, min_child_weight = 3, random_state = 42, n_jobs = 4)
}

def print_row(name, m):
    print(f"{name: <38} n {m['n']: >4} exact {m['exact']: 6.1%} within_1 {m['within_1']: 6.1%}  "
          f"MAE {m['mae']: .2f} QWK {m['qwk']: .3f}")
    
def main():
    coco, gt, pred = load_routes()
    lookup = build_owner_lookup(gt)
    
    gt_table = build_table(coco, gt)
    add_labels(gt_table, *lookup)
    gt_table = gt_table[gt_table["y"].notna()].reset_index(drop = True) # drop INC routes
    gt_table["y"] = gt_table["y"].astype(int)
    
    pred_table = build_table(coco, pred)
    add_labels(pred_table, *lookup)
    
    features = feature_columns(gt_table)
    print(f"{len(gt_table)} graded routes, {gt_table['wall'].nunique()} walls, {len(features)} features\n")
    
    results = {}
    for name, make_model in MODELS.items():
        gt_raw, pred_raw = leave_one_wall_out(make_model, gt_table, features, pred_table)
        graded = pred_table["y"].notna() & pred_raw.notna()
        matched = graded & (pred_table["iou"] >= IOU_MATCH)
        
        results[name] = {
            "gt": score(gt_table["y"], gt_raw),
            "phase2_all": score(pred_table.loc[graded, "y"], pred_raw[graded]),
            "phase2_matched": score(pred_table.loc[matched, "y"], pred_raw[matched])
        }
        
        print_row(f"{name}: labelled routes", results[name]["gt"])
        print_row(f"{name}: phase 2 routes", results[name]["phase2_all"])
        print_row(f"{name}: phase 2, IoU >= {IOU_MATCH}", results[name]["phase2_matched"])
        
        print()
        
        gt_table[f"pred_{name}"] = gt_raw
        pred_table[f"pred_{name}"] = pred_raw
        
    best = "xgboost"
    cm = confusion_matrix(gt_table["y"], to_grade(gt_table[f"pred_{best}"]), labels = range(len(GRADES)))
    
    print(f"{best} confusion (rows = true, cols = predicted)")
    print("      " + "".join(f"{g: >5}" for g in GRADES))
    
    for grade, row in zip(GRADES, cm):
        print(f"{grade: >5} " + " ".join(f"{v: >5}" for v in row))
        
    final = MODELS[best]().fit(gt_table[features], gt_table["y"]) # all wall, for interpretation only
    importance = pd.Series(final.feature_importances_, index = features).sort_values(ascending = False)
    
    print("\ntop features:\n" + importance.head(10).to_string())
    
    os.makedirs(OUT_DIR, exist_ok = True)
    gt_table.drop(columns = features).to_csv(f"{OUT_DIR}/gt_route_grades.csv", index = False)
    pred_table.drop(columns = features).to_csv(f"{OUT_DIR}/predicted_route_grades.csv", index = False)
    
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({"grades": GRADES, "results": results, "confusion": cm.tolist(),
                  "importance": importance.to_dict()}, f, indent = 1)
        
if __name__ == "__main__":
    main()