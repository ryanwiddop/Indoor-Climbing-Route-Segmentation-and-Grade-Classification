"""
Evaluate the phase 1 detector on whole walls against a COCO ground truth.

Matching is CLASS-AWARE: a detection counts as a true positive only when its
predicted hold type equals the GT type and box IoU clears the threshold. mAP is
the mean of per-class APs, so a class the model never gets right drags the mean
down even if localization is perfect.

By default only the held-out val walls are scored, using the same wall-level
split as prepare_tiles.py (seed 42). Volumes stay in every pool but are also
reported on their own, since they are much larger than holds and otherwise
flatter the mean IoU.

Usage:
    python phase_1_eval.py [--walls {val,train,all}] [options]
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
for _p in (HERE, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase_1 import box_iou, load_model, tiled_predict
from datasets.whole_wall_dataset import WholeWallDataset, collate_fn
from data_preprocessing.prepare_tiles import split_walls

DEFAULT_CKPT = os.path.join(HERE, "checkpoints/phase_1.pt")
DEFAULT_IMG_DIR = "/home/public/rwiddop/images/"
DEFAULT_COCO = "/home/public/rwiddop/annotations.json"
DEFAULT_FIG_DIR = os.path.join(HERE, "figures/phase_1/eval")
DEFAULT_LOG_DIR = os.path.join(HERE, "logs")
LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "phase_1_eval.log")

VOLUME_NAME = "volume"

logger = logging.getLogger("phase_1_eval")


def setup_logger():
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        for h in (logging.FileHandler(LOG_FILE), logging.StreamHandler()):
            h.setLevel(logging.INFO)
            h.setFormatter(fmt)
            logger.addHandler(h)
        logger.propagate = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    p.add_argument("--coco-json", default=DEFAULT_COCO)
    p.add_argument("--output-dir", default=DEFAULT_FIG_DIR)
    p.add_argument("--walls", choices=["val", "train", "all"], default="val",
                   help="Which wall split to score (default: val)")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Must match the value prepare_tiles.py was run with")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the value prepare_tiles.py was run with")
    p.add_argument("--score-threshold", type=float, default=0.6)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--max-detections", type=int, default=500)
    p.add_argument("--tile-size", type=int, default=1600)
    p.add_argument("--tile-overlap", type=float, default=0.66)
    p.add_argument("--tile-score-threshold", type=float, default=0.05)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--device", default=None)
    return p.parse_args()


def greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thresh):
    n_pred = len(pred_boxes)
    matches = [-1] * n_pred
    ious = [0.0] * n_pred
    if n_pred == 0 or len(gt_boxes) == 0:
        return matches, ious
    order = np.argsort(-np.asarray(pred_scores))
    matched_gt = set()
    for i in order:
        best_iou, best_j = 0.0, -1
        pb = pred_boxes[i]
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j != -1:
            matches[i] = best_j
            ious[i] = best_iou
            matched_gt.add(best_j)
    return matches, ious


def greedy_match_classaware(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresh):
    """Greedy IoU matching restricted to same-class pairs.

    Returns GT indices into the full per-image GT arrays, so callers can use a
    match to index polygons and labels directly.
    """
    n_pred = len(pred_boxes)
    matches = [-1] * n_pred
    ious = [0.0] * n_pred
    if n_pred == 0 or len(gt_boxes) == 0:
        return matches, ious
    order = np.argsort(-np.asarray(pred_scores))
    matched_gt = set()
    for i in order:
        best_iou, best_j = 0.0, -1
        pb = pred_boxes[i]
        pl = pred_labels[i]
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt or gt_labels[j] != pl:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j != -1:
            matches[i] = best_j
            ious[i] = best_iou
            matched_gt.add(best_j)
    return matches, ious


def nanmean(values):
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def compute_ap(sorted_tp_flags, num_gt):
    if num_gt == 0:
        return float("nan")
    if len(sorted_tp_flags) == 0:
        return 0.0
    tp = np.cumsum(sorted_tp_flags)
    fp = np.cumsum(1 - np.asarray(sorted_tp_flags))
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pr_curve_for_class(per_image, class_id, iou_thresh):
    scores, tps = [], []
    for r in per_image:
        pm = r["pred_labels"] == class_id
        gm = r["gt_labels"] == class_id
        pb, ps = r["pred_boxes"][pm], r["pred_scores"][pm]
        gb = r["gt_boxes"][gm]
        matches, _ = greedy_match(pb, ps, gb, iou_thresh)
        scores.extend(ps.tolist())
        tps.extend([1 if m != -1 else 0 for m in matches])
    num_gt = sum(int((r["gt_labels"] == class_id).sum()) for r in per_image)
    if not scores:
        return np.array([]), np.array([]), num_gt, np.array([])
    order = np.argsort(-np.asarray(scores))
    sorted_tp = np.asarray(tps)[order]
    tp_cum = np.cumsum(sorted_tp)
    fp_cum = np.cumsum(1 - sorted_tp)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return recall, precision, num_gt, sorted_tp


def compute_per_class_ap(per_image, iou_thresholds, class_ids):
    """{class_id: {iou_threshold: AP}}; AP is NaN for a class with no GT."""
    out = {}
    for c in class_ids:
        num_gt = sum(int((r["gt_labels"] == c).sum()) for r in per_image)
        per_t = {}
        for t in iou_thresholds:
            _, _, _, sorted_tp = pr_curve_for_class(per_image, c, t)
            per_t[t] = compute_ap(sorted_tp, num_gt)
        out[c] = per_t
    return out


def mask_iou_local(pred_mask_entry, gt_polygon, pred_box, img_size, pad=8):
    W, H = img_size
    x_off, y_off, local_mask = pred_mask_entry
    xs, ys = gt_polygon

    x1 = max(0, int(min(min(xs), pred_box[0])) - pad)
    y1 = max(0, int(min(min(ys), pred_box[1])) - pad)
    x2 = min(W, int(max(max(xs), pred_box[2])) + pad)
    y2 = min(H, int(max(max(ys), pred_box[3])) + pad)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi_w, roi_h = x2 - x1, y2 - y1

    gt_img = Image.new("L", (roi_w, roi_h), 0)
    ImageDraw.Draw(gt_img).polygon(
        list(zip([x - x1 for x in xs], [y - y1 for y in ys])), fill=1
    )
    gt_arr = np.array(gt_img, dtype=bool)

    pred_roi = np.zeros((roi_h, roi_w), dtype=bool)
    if local_mask.size > 0:
        mh, mw = local_mask.shape
        ox1 = max(x_off, x1); oy1 = max(y_off, y1)
        ox2 = min(x_off + mw, x2); oy2 = min(y_off + mh, y2)
        if ox2 > ox1 and oy2 > oy1:
            lx1, ly1 = ox1 - x_off, oy1 - y_off
            lx2, ly2 = ox2 - x_off, oy2 - y_off
            rx1, ry1 = ox1 - x1, oy1 - y1
            rx2, ry2 = ox2 - x1, oy2 - y1
            pred_roi[ry1:ry2, rx1:rx2] = local_mask[ly1:ly2, lx1:lx2]

    inter = np.logical_and(gt_arr, pred_roi).sum()
    union = np.logical_or(gt_arr, pred_roi).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def visualize_wall(image_pil, gt_boxes, gt_labels, gt_polygons, pred_boxes, pred_labels,
                   pred_scores, pred_masks, idx_to_name, score_threshold, output_path, title=""):
    base = image_pil.convert("RGBA").copy()
    W, H = base.size

    alpha_plane = np.zeros((H, W), dtype=np.uint8)
    for (x_off, y_off, local_mask), score in zip(pred_masks, pred_scores):
        if score < score_threshold or local_mask.size == 0:
            continue
        mh, mw = local_mask.shape
        x2 = min(W, x_off + mw); y2 = min(H, y_off + mh)
        if x2 <= x_off or y2 <= y_off:
            continue
        sub = alpha_plane[y_off:y2, x_off:x2]
        np.maximum(sub, (local_mask[: y2 - y_off, : x2 - x_off].astype(np.uint8) * 160), out=sub)

    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(Image.fromarray(alpha_plane))
    overlay = Image.alpha_composite(Image.new("RGBA", base.size, (0, 0, 0, 0)), red)
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    for xs, ys in gt_polygons:
        draw.polygon(list(zip(xs, ys)), outline=(0, 255, 0, 255))
    for bx, lb in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [int(v) for v in bx]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)
        draw.text((x1, max(0, y1 - 18)), idx_to_name.get(int(lb), str(lb)),
                  fill=(0, 255, 0, 255), font=font)

    for bx, lb, sc in zip(pred_boxes, pred_labels, pred_scores):
        if sc < score_threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in bx]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
        draw.text((x1, min(y2 + 2, H - 18)),
                  f"{idx_to_name.get(int(lb), str(lb))} {sc:.2f}",
                  fill=(255, 0, 0, 255), font=font)

    composited = Image.alpha_composite(base, overlay).convert("RGB")

    plt.figure(figsize=(16, 12))
    plt.imshow(composited)
    plt.title(title + "  |  green=GT, red=pred")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def select_walls(coco_json, which, val_frac, seed):
    with open(coco_json) as f:
        raw = json.load(f)
    all_names = [im["file_name"] for im in raw["images"]]
    train_walls, val_walls = split_walls(all_names, val_frac, seed)
    if which == "val":
        return set(val_walls), len(all_names)
    if which == "train":
        return set(train_walls), len(all_names)
    return set(all_names), len(all_names)


def summarise_group(stats, class_ids):
    tp = sum(stats["tp"][c] for c in class_ids)
    fp = sum(stats["fp"][c] for c in class_ids)
    fn = sum(stats["fn"][c] for c in class_ids)
    box_ious, mask_ious = [], []
    for c in class_ids:
        box_ious.extend(stats["box_ious"][c])
        mask_ious.extend(stats["mask_ious"][c])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mean_matched_box_iou": float(np.mean(box_ious)) if box_ious else None,
        "mean_matched_mask_iou": float(np.mean(mask_ious)) if mask_ious else None,
        "n_matched": len(box_ious),
    }


def main():
    setup_logger()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model, hold_type_to_idx, num_classes = load_model(
        args.checkpoint, device, args.max_detections,
    )
    idx_to_name = {v: k for k, v in hold_type_to_idx.items()}

    wanted_walls, n_total_walls = select_walls(
        args.coco_json, args.walls, args.val_frac, args.seed)

    logger.info("=" * 80)
    logger.info("PHASE 1 EVAL (whole-wall, class-aware) | pid=%s", os.getpid())
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Image dir : %s", args.img_dir)
    logger.info("COCO GT   : %s", args.coco_json)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Walls     : %s split (%d of %d) seed=%d val_frac=%.2f",
                args.walls, len(wanted_walls), n_total_walls, args.seed, args.val_frac)
    logger.info("            %s", sorted(wanted_walls))
    logger.info("Inference : Tiled | Tile=%d  overlap=%.2f  nms_iou=%.2f  tile_score>=%.2f",
                args.tile_size, args.tile_overlap, args.nms_iou, args.tile_score_threshold)
    logger.info("Op point  : score>=%.2f  IoU>=%.2f  max_dets=%d",
                args.score_threshold, args.iou_threshold, args.max_detections)
    logger.info("Device    : %s", device)
    logger.info("Classes   : %s", hold_type_to_idx)
    logger.info("=" * 80)

    dataset = WholeWallDataset(
        args.img_dir, args.coco_json, hold_type_to_idx=hold_type_to_idx, return_masks=False)

    gt_cat_names = {c["name"] for c in dataset.coco.loadCats(dataset.coco.getCatIds())}
    missing = sorted(gt_cat_names - set(hold_type_to_idx))
    if missing:
        logger.error(
            "GT categories %s are absent from the checkpoint's classes %s. Their "
            "annotations are dropped by the dataset, which inflates recall. "
            "The checkpoint and the COCO disagree - retrain or re-export.",
            missing, sorted(hold_type_to_idx),
        )

    dataset.image_ids = [
        i for i in dataset.image_ids
        if dataset.coco.imgs[i]["file_name"] in wanted_walls
    ]
    found = {dataset.coco.imgs[i]["file_name"] for i in dataset.image_ids}
    absent = sorted(wanted_walls - found)
    if absent:
        logger.warning("Requested walls absent from COCO or image dir: %s", absent)
    if len(dataset) == 0:
        logger.error("No walls to evaluate. Aborting.")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    n_gt = sum(len(dataset.coco.getAnnIds(imgIds=i)) for i in dataset.image_ids)
    logger.info("Dataset: %d walls, %d GT annotations", len(dataset), n_gt)

    per_image = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating walls", unit="wall"):
            if not images:
                continue
            image = images[0]
            target = targets[0]
            W, H = target["image_size"]

            pred = tiled_predict(
                model, image, device,
                tile_size=args.tile_size, overlap=args.tile_overlap,
                score_threshold=args.tile_score_threshold,
                nms_iou=args.nms_iou,
                volume_label=hold_type_to_idx.get(VOLUME_NAME),
            )

            per_image.append({
                "filename": target["filename"], "image_size": (W, H),
                "gt_boxes": target["boxes"].numpy(),
                "gt_labels": target["labels"].numpy(),
                "gt_polygons": target["polygons"],
                "pred_boxes":  pred["boxes"],
                "pred_scores": pred["scores"],
                "pred_labels": pred["labels"],
                "pred_masks":  pred["masks"],
                "image_tensor": image,
            })

    logger.info("Inference done in %.1fs across %d walls", time.perf_counter() - t0, len(per_image))

    class_ids = sorted(idx_to_name)
    volume_id = hold_type_to_idx.get(VOLUME_NAME)
    hold_ids = [c for c in class_ids if c != volume_id]

    stats = {
        "tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int),
        "box_ious": defaultdict(list), "mask_ious": defaultdict(list),
        "score_tp": [], "score_fp": [],
    }
    per_img_stats = []

    # Class-aware matching throws away the pred<->GT pairing whenever the types
    # disagree, so it cannot say WHICH types get confused. The confusion matrix
    # therefore comes from a second, localization-only pass.
    n_cls = len(class_ids)
    pos = {c: k for k, c in enumerate(class_ids)}
    NONE_IDX = n_cls
    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)

    for rec in per_image:
        keep = rec["pred_scores"] >= args.score_threshold
        pb = rec["pred_boxes"][keep]
        ps = rec["pred_scores"][keep]
        pl = rec["pred_labels"][keep]
        pm = [rec["pred_masks"][i] for i, k in enumerate(keep.tolist()) if k]

        gb = rec["gt_boxes"]
        gl = rec["gt_labels"]

        matches, ious = greedy_match_classaware(pb, ps, pl, gb, gl, args.iou_threshold)
        matched_gt = {m for m in matches if m != -1}

        for i, m in enumerate(matches):
            c = int(pl[i])
            if m != -1:
                stats["tp"][c] += 1
                stats["box_ious"][c].append(ious[i])
                stats["mask_ious"][c].append(
                    mask_iou_local(pm[i], rec["gt_polygons"][m], pb[i], rec["image_size"]))
                stats["score_tp"].append(float(ps[i]))
            else:
                stats["fp"][c] += 1
                stats["score_fp"].append(float(ps[i]))

        for j in range(len(gb)):
            if j not in matched_gt:
                stats["fn"][int(gl[j])] += 1

        loc_matches, _ = greedy_match(pb, ps, gb, args.iou_threshold)
        loc_matched_gt = {m for m in loc_matches if m != -1}
        for i, m in enumerate(loc_matches):
            if m != -1:
                cm[pos[int(gl[m])], pos[int(pl[i])]] += 1
            else:
                cm[NONE_IDX, pos[int(pl[i])]] += 1
        for j in range(len(gb)):
            if j not in loc_matched_gt:
                cm[pos[int(gl[j])], NONE_IDX] += 1

        n_tp = len(matched_gt)
        per_img_stats.append({
            "filename": rec["filename"],
            "gt_count": int(len(gb)),
            "pred_count_raw": int(len(rec["pred_boxes"])),
            "pred_count_kept": int(keep.sum()),
            "tp": int(n_tp),
            "fp": int(len(matches) - n_tp),
            "fn": int(len(gb) - n_tp),
            "recall": n_tp / max(len(gb), 1),
            "precision": n_tp / max(len(matches), 1),
        })

    overall = summarise_group(stats, class_ids)
    holds_only = summarise_group(stats, hold_ids)
    volume_only = summarise_group(stats, [volume_id]) if volume_id is not None else None

    logger.info("-" * 72)
    logger.info("Operating point: score>=%.2f  IoU>=%.2f  (class-aware matching)",
                args.score_threshold, args.iou_threshold)
    for name, grp in (("ALL classes", overall), ("holds only", holds_only),
                      ("volume only", volume_only)):
        if grp is None:
            continue
        logger.info("  %-12s TP=%-5d FP=%-5d FN=%-5d  P=%.4f R=%.4f F1=%.4f  boxIoU=%s maskIoU=%s",
                    name, grp["TP"], grp["FP"], grp["FN"],
                    grp["precision"], grp["recall"], grp["f1"],
                    f"{grp['mean_matched_box_iou']:.4f}" if grp["mean_matched_box_iou"] is not None else "n/a",
                    f"{grp['mean_matched_mask_iou']:.4f}" if grp["mean_matched_mask_iou"] is not None else "n/a")

    loc_tp = int(cm[:n_cls, :n_cls].sum())
    loc_fp = int(cm[NONE_IDX, :n_cls].sum())
    loc_fn = int(cm[:n_cls, NONE_IDX].sum())
    loc_precision = loc_tp / max(loc_tp + loc_fp, 1)
    loc_recall = loc_tp / max(loc_tp + loc_fn, 1)
    loc_f1 = 2 * loc_precision * loc_recall / max(loc_precision + loc_recall, 1e-12)
    type_correct = int(np.trace(cm[:n_cls, :n_cls]))
    type_accuracy = type_correct / max(loc_tp, 1)

    logger.info("-" * 72)
    logger.info("Localization only (class-agnostic, same IoU gate):")
    logger.info("  found=%-5d spurious=%-5d missed=%-5d  P=%.4f R=%.4f F1=%.4f",
                loc_tp, loc_fp, loc_fn, loc_precision, loc_recall, loc_f1)
    logger.info("  type accuracy on found holds: %d/%d = %.4f",
                type_correct, loc_tp, type_accuracy)

    names = [idx_to_name[c] for c in class_ids]
    head = "".join(f"{n[:6]:>7}" for n in names)
    logger.info("-" * 72)
    logger.info("Confusion matrix (rows=GT, cols=predicted; localization-matched pairs)")
    logger.info("%-8s%s%7s", "", head, "MISSED")
    for r, c in enumerate(class_ids):
        row = "".join(f"{cm[r, k]:>7d}" for k in range(n_cls))
        logger.info("%-8s%s%7d", names[r][:7], row, cm[r, NONE_IDX])
    logger.info("%-8s%s%7s", "SPURIOUS",
                "".join(f"{cm[NONE_IDX, k]:>7d}" for k in range(n_cls)), "-")

    iou_range_coco = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    per_class_ap = compute_per_class_ap(per_image, iou_range_coco, class_ids)

    def mean_over_classes(ids, t=None):
        if t is None:
            return nanmean([nanmean([per_class_ap[c][x] for x in iou_range_coco]) for c in ids])
        return nanmean([per_class_ap[c][t] for c in ids])

    scored_ids = [c for c in class_ids
                  if sum(int((r["gt_labels"] == c).sum()) for r in per_image) > 0]
    map50 = mean_over_classes(class_ids, 0.5)
    map75 = mean_over_classes(class_ids, 0.75)
    map_coco = mean_over_classes(class_ids)

    logger.info("-" * 72)
    logger.info("%-10s %8s %8s %8s %8s", "class", "n_GT", "AP@.50", "AP@.75", "AP@[.5:.95]")
    for c in class_ids:
        n_gt_c = sum(int((r["gt_labels"] == c).sum()) for r in per_image)
        ap_all = nanmean([per_class_ap[c][t] for t in iou_range_coco])
        logger.info("%-10s %8d %8.4f %8.4f %8.4f", idx_to_name[c], n_gt_c,
                    per_class_ap[c][0.5], per_class_ap[c][0.75], ap_all)
    logger.info("-" * 72)
    logger.info("mAP@0.50     = %.4f", map50)
    logger.info("mAP@0.75     = %.4f", map75)
    logger.info("mAP@[.5:.95] = %.4f   (mean over %d classes with GT; %d skipped)",
                map_coco, len(scored_ids), len(class_ids) - len(scored_ids))
    logger.info("mAP@[.5:.95] holds only  = %.4f", mean_over_classes(hold_ids))
    if volume_id is not None:
        logger.info("mAP@[.5:.95] volume only = %.4f", mean_over_classes([volume_id]))
    logger.info("-" * 72)

    figdir = args.output_dir

    logger.info("Saving per-image overlays...")
    for rec in tqdm(per_image, desc="Viz", unit="wall"):
        img_np = rec["image_tensor"].permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        out_path = os.path.join(
            figdir, f"phase_1_eval_pred_{os.path.splitext(rec['filename'])[0]}.png")
        visualize_wall(
            img_pil,
            rec["gt_boxes"], rec["gt_labels"], rec["gt_polygons"],
            rec["pred_boxes"], rec["pred_labels"], rec["pred_scores"], rec["pred_masks"],
            idx_to_name,
            score_threshold=args.score_threshold,
            output_path=out_path,
            title=(f"{rec['filename']}  GT={len(rec['gt_boxes'])}  "
                   f"Pred(>={args.score_threshold})={int((rec['pred_scores'] >= args.score_threshold).sum())}"),
        )

    plt.figure(figsize=(9, 7))
    for c in class_ids:
        recall, precision, n_gt_c, _ = pr_curve_for_class(per_image, c, args.iou_threshold)
        if n_gt_c == 0 or len(recall) == 0:
            continue
        plt.plot(recall, precision, linewidth=1.8,
                 label=f"{idx_to_name[c]} (AP={per_class_ap[c][args.iou_threshold]:.3f}, n={n_gt_c})"
                 if args.iou_threshold in per_class_ap[c] else idx_to_name[c])
    plt.xlim(0, 1); plt.ylim(0, 1.02)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Per-class PR @ IoU={args.iou_threshold:.2f}   mAP50={map50:.3f}")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_pr_curve.png"), dpi=180)
    plt.close()

    cm_labels = names + ["(missed)"]
    cm_rows = np.vstack([cm[:n_cls, :], cm[NONE_IDX:NONE_IDX + 1, :]])
    row_tot = cm_rows.sum(axis=1, keepdims=True)
    cm_norm = cm_rows / np.maximum(row_tot, 1)
    plt.figure(figsize=(max(9, n_cls * 1.2), max(8, n_cls * 1.1)))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(label="fraction of GT row")
    plt.xticks(np.arange(n_cls + 1), cm_labels, rotation=45, ha="right")
    plt.yticks(np.arange(n_cls + 1), names + ["(spurious)"])
    for i in range(cm_rows.shape[0]):
        for j in range(cm_rows.shape[1]):
            plt.text(j, i, str(cm_rows[i, j]), ha="center", va="center", fontsize=9,
                     color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.xlabel("Predicted type")
    plt.ylabel("GT type")
    plt.title(f"Hold-type confusion on localization-matched pairs (IoU>={args.iou_threshold:.2f})\n"
              f"type accuracy = {type_correct}/{loc_tp} = {type_accuracy:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_confusion_matrix.png"), dpi=180)
    plt.close()

    ap_vals = [nanmean([per_class_ap[c][t] for t in iou_range_coco]) for c in class_ids]
    gt_counts_cls = [sum(int((r["gt_labels"] == c).sum()) for r in per_image) for c in class_ids]
    plt.figure(figsize=(max(8, len(names) * 1.1), 6))
    bars = plt.bar(names, ap_vals, color="#1f77b4", edgecolor="black")
    plt.axhline(map_coco, color="black", linestyle="--", label=f"mAP={map_coco:.3f}")
    plt.ylim(0, 1.05); plt.ylabel("AP@[.5:.95]")
    plt.title("Per-class AP")
    for b, v, n in zip(bars, ap_vals, gt_counts_cls):
        plt.text(b.get_x() + b.get_width() / 2, (v if not np.isnan(v) else 0) + 0.01,
                 f"{v:.2f}\nn={n}" if not np.isnan(v) else f"n={n}",
                 ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_per_class_ap.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    bins = np.linspace(0, 1, 21)
    plt.hist([stats["score_tp"], stats["score_fp"]], bins=bins, stacked=True,
             label=[f"TP (n={len(stats['score_tp'])})", f"FP (n={len(stats['score_fp'])})"],
             color=["#2ca02c", "#d62728"], edgecolor="black")
    plt.axvline(args.score_threshold, color="black", linestyle="--",
                label=f"threshold={args.score_threshold}")
    plt.xlabel("Predicted score"); plt.ylabel("Count")
    plt.title("Prediction score distribution (class-aware TP/FP, at operating point)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_score_hist.png"), dpi=180)
    plt.close()

    all_box_ious = [v for c in class_ids for v in stats["box_ious"][c]]
    all_mask_ious = [v for c in class_ids for v in stats["mask_ious"][c]]

    if all_box_ious:
        plt.figure(figsize=(9, 6))
        hold_vals = [v for c in hold_ids for v in stats["box_ious"][c]]
        vol_vals = stats["box_ious"][volume_id] if volume_id is not None else []
        plt.hist([hold_vals, vol_vals], bins=np.linspace(args.iou_threshold, 1, 20),
                 stacked=True, label=[f"holds (n={len(hold_vals)})", f"volume (n={len(vol_vals)})"],
                 color=["#1f77b4", "#ff7f0e"], edgecolor="black")
        plt.xlabel("Box IoU"); plt.ylabel("Count")
        plt.title(f"Box-IoU of matched predictions  (mean={np.mean(all_box_ious):.3f})")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figdir, "phase_1_eval_box_iou_hist.png"), dpi=180)
        plt.close()

    if all_mask_ious:
        plt.figure(figsize=(9, 6))
        hold_vals = [v for c in hold_ids for v in stats["mask_ious"][c]]
        vol_vals = stats["mask_ious"][volume_id] if volume_id is not None else []
        plt.hist([hold_vals, vol_vals], bins=np.linspace(0, 1, 20),
                 stacked=True, label=[f"holds (n={len(hold_vals)})", f"volume (n={len(vol_vals)})"],
                 color=["#9467bd", "#ff7f0e"], edgecolor="black")
        plt.xlabel("Mask IoU"); plt.ylabel("Count")
        plt.title(f"Mask-IoU of matched predictions  (mean={np.mean(all_mask_ious):.3f})")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figdir, "phase_1_eval_mask_iou_hist.png"), dpi=180)
        plt.close()

    fnames = [s["filename"] for s in per_img_stats]
    recalls = [s["recall"] for s in per_img_stats]
    plt.figure(figsize=(max(8, len(fnames) * 0.9), 6))
    bars = plt.bar(fnames, recalls, color="#17becf", edgecolor="black")
    plt.axhline(overall["recall"], color="black", linestyle="--",
                label=f"overall recall={overall['recall']:.3f}")
    plt.ylim(0, 1.05); plt.ylabel("Recall")
    plt.title(f"Per-image recall @ score>={args.score_threshold}, IoU>={args.iou_threshold}")
    plt.xticks(rotation=45, ha="right")
    for b, v in zip(bars, recalls):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_per_image_recall.png"), dpi=180)
    plt.close()

    gt_counts = [s["gt_count"] for s in per_img_stats]
    pred_counts = [s["pred_count_kept"] for s in per_img_stats]
    x = np.arange(len(fnames)); width = 0.38
    plt.figure(figsize=(max(8, len(fnames) * 0.9), 6))
    plt.bar(x - width / 2, gt_counts,   width, label="GT", color="#2ca02c", edgecolor="black")
    plt.bar(x + width / 2, pred_counts, width, label=f"Pred (>={args.score_threshold})",
            color="#d62728", edgecolor="black")
    plt.xticks(x, fnames, rotation=45, ha="right")
    plt.ylabel("Count"); plt.title("GT vs kept predictions per wall")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_gt_vs_pred_counts.png"), dpi=180)
    plt.close()

    metrics = {
        "operating_point": {
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "matching": "class-aware",
            "all_classes": overall,
            "holds_only": holds_only,
            "volume_only": volume_only,
        },
        "localization_only": {
            "matching": "class-agnostic",
            "found": loc_tp, "spurious": loc_fp, "missed": loc_fn,
            "precision": float(loc_precision), "recall": float(loc_recall), "f1": float(loc_f1),
            "type_accuracy_on_found": float(type_accuracy),
            "type_correct": type_correct,
        },
        "confusion_matrix": {
            "note": "rows=GT type, cols=predicted type, from localization-only matching",
            "classes": names,
            "row_missed": "GT of this type matched no detection",
            "col_spurious": "detection of this type matched no GT",
            "matrix": cm[:n_cls, :n_cls].tolist(),
            "missed_per_gt_class": cm[:n_cls, NONE_IDX].tolist(),
            "spurious_per_pred_class": cm[NONE_IDX, :n_cls].tolist(),
        },
        "mAP": {
            "mAP@0.50": map50,
            "mAP@0.75": map75,
            "mAP@[.5:.95]": map_coco,
            "mAP@[.5:.95]_holds_only": mean_over_classes(hold_ids),
            "mAP@[.5:.95]_volume_only": (
                mean_over_classes([volume_id]) if volume_id is not None else None),
            "per_class": {
                idx_to_name[c]: {
                    "n_gt": sum(int((r["gt_labels"] == c).sum()) for r in per_image),
                    "AP@0.50": per_class_ap[c][0.5],
                    "AP@0.75": per_class_ap[c][0.75],
                    "AP@[.5:.95]": nanmean([per_class_ap[c][t] for t in iou_range_coco]),
                    "per_iou_threshold": {f"{t:.2f}": per_class_ap[c][t] for t in iou_range_coco},
                    "TP": int(stats["tp"][c]), "FP": int(stats["fp"][c]), "FN": int(stats["fn"][c]),
                }
                for c in class_ids
            },
        },
        "per_image": per_img_stats,
        "config": {
            "checkpoint": args.checkpoint,
            "img_dir": args.img_dir,
            "coco_json": args.coco_json,
            "walls": args.walls,
            "wall_list": sorted(found),
            "val_frac": args.val_frac,
            "seed": args.seed,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "nms_iou": args.nms_iou,
            "tile_score_threshold": args.tile_score_threshold,
            "max_detections_per_img": args.max_detections,
            "device": str(device),
            "num_classes": num_classes,
            "hold_type_to_idx": hold_type_to_idx,
        },
    }
    metrics_path = os.path.join(figdir, "phase_1_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", metrics_path)
    logger.info("Figures written to %s", figdir)


if __name__ == "__main__":
    main()
