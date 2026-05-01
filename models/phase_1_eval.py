import argparse
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase_1 import box_iou, load_model, tiled_predict
from datasets.whole_wall_dataset import WholeWallDataset, collate_fn

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints/phase_1.pt")
DEFAULT_IMG_DIR = "/home/public/rwiddop/images/"
DEFAULT_ANN_CSV = "/home/public/rwiddop/annotation.csv"
DEFAULT_FIG_DIR = os.path.join(HERE, "figures/phase_1/eval")
DEFAULT_LOG_DIR = os.path.join(HERE, "logs")
LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "phase_1_eval.log")

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
    p.add_argument("--ann-csv", default=DEFAULT_ANN_CSV)
    p.add_argument("--output-dir", default=DEFAULT_FIG_DIR)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--max-detections", type=int, default=500)
    p.add_argument("--tile-size", type=int, default=800)
    p.add_argument("--tile-overlap", type=float, default=0.25)
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


def compute_map(pred_boxes_per_img, pred_scores_per_img, gt_boxes_per_img, iou_thresholds):
    results = {}
    num_gt_total = sum(len(g) for g in gt_boxes_per_img)
    for t in iou_thresholds:
        all_scores, all_tp = [], []
        for pb, ps, gb in zip(pred_boxes_per_img, pred_scores_per_img, gt_boxes_per_img):
            matches, _ = greedy_match(pb, ps, gb, t)
            all_scores.extend(ps.tolist())
            all_tp.extend([1 if m != -1 else 0 for m in matches])
        if all_scores:
            order = np.argsort(-np.asarray(all_scores))
            sorted_tp = np.asarray(all_tp)[order]
        else:
            sorted_tp = np.array([])
        results[t] = compute_ap(sorted_tp, num_gt_total)
    return results


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


def visualize_wall(image_pil, gt_boxes, gt_polygons, pred_boxes, pred_scores, pred_masks, score_threshold, output_path, title=""):
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

    for xs, ys in gt_polygons:
        draw.polygon(list(zip(xs, ys)), outline=(0, 255, 0, 255))
    for bx in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in bx]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
    for bx, sc in zip(pred_boxes, pred_scores):
        if sc < score_threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in bx]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
        draw.text((x1, max(0, y1 - 18)), f"{sc:.2f}", fill=(255, 0, 0, 255), font=font)

    composited = Image.alpha_composite(base, overlay).convert("RGB")

    plt.figure(figsize=(16, 12))
    plt.imshow(composited)
    plt.title(title + "  |  green=GT, red=pred")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    setup_logger()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 1 EVAL (whole-wall) | pid=%s", os.getpid())
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Image dir : %s", args.img_dir)
    logger.info("Ann CSV   : %s", args.ann_csv)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Inference : Tiled | Tile=%d  overlap=%.2f  nms_iou=%.2f  tile_score>=%.2f", args.tile_size, args.tile_overlap, args.nms_iou, args.tile_score_threshold)
    logger.info("Op point  : score>=%.2f  IoU>=%.2f  max_dets=%d", args.score_threshold, args.iou_threshold, args.max_detections)
    logger.info("=" * 80)

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logger.info("Using device: %s", device)

    model, hold_type_to_idx, num_classes = load_model(
        args.checkpoint, device, args.max_detections,
    )
    logger.info("Loaded model. Classes: %s", hold_type_to_idx)

    dataset = WholeWallDataset(args.img_dir, args.ann_csv, hold_type_to_idx=hold_type_to_idx, return_masks=False)
    if len(dataset) == 0:
        logger.error("No wall images found in %s. Aborting.", args.img_dir)
        return
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    logger.info("Dataset: %d walls, %d GT annotations total", len(dataset), sum(len(dataset.rows_by_img[f]) for f in dataset.img_files))

    per_image = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating walls", unit="wall"):
            if not images:
                continue
            image = images[0]
            target = targets[0]
            W, H = target["image_size"]
            fname = target["filename"]

            pred = tiled_predict(
                model, image, device,
                tile_size=args.tile_size, overlap=args.tile_overlap,
                score_threshold=args.tile_score_threshold,
                nms_iou=args.nms_iou,
            )

            per_image.append({
                "filename": fname, "image_size": (W, H),
                "gt_boxes": target["boxes"].numpy(),
                "gt_polygons": target["polygons"],
                "pred_boxes":  pred["boxes"],
                "pred_scores": pred["scores"],
                "pred_labels": pred["labels"],
                "pred_masks":  pred["masks"],
                "image_tensor": image,
            })

    logger.info("Inference done in %.1fs across %d walls", time.perf_counter() - t0, len(per_image))

    tp_total = fp_total = fn_total = 0
    matched_box_ious, matched_mask_ious = [], []
    score_tp, score_fp = [], []
    per_img_stats = []

    for rec in per_image:
        keep = rec["pred_scores"] >= args.score_threshold
        pb = rec["pred_boxes"][keep]
        ps = rec["pred_scores"][keep]

        pm_all = rec["pred_masks"]
        pm = [pm_all[i] for i, k in enumerate(keep.tolist()) if k]

        matches, ious = greedy_match(pb, ps, rec["gt_boxes"], args.iou_threshold)
        n_tp = sum(1 for m in matches if m != -1)
        n_fp = len(matches) - n_tp
        n_fn = len(rec["gt_boxes"]) - n_tp
        tp_total += n_tp; fp_total += n_fp; fn_total += n_fn

        for i, m in enumerate(matches):
            if m != -1:
                matched_box_ious.append(ious[i])
                score_tp.append(float(ps[i]))
                m_iou = mask_iou_local(
                    pm[i], rec["gt_polygons"][m], pb[i], rec["image_size"]
                )
                matched_mask_ious.append(m_iou)
            else:
                score_fp.append(float(ps[i]))

        per_img_stats.append({
            "filename": rec["filename"],
            "gt_count": len(rec["gt_boxes"]),
            "pred_count_raw": len(rec["pred_boxes"]),
            "pred_count_kept": int(keep.sum()),
            "tp": n_tp, "fp": n_fp, "fn": n_fn,
            "recall":    n_tp / max(len(rec["gt_boxes"]), 1),
            "precision": n_tp / max(len(matches), 1),
        })

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    logger.info("-" * 60)
    logger.info("Op point: score>=%.2f  IoU>=%.2f", args.score_threshold, args.iou_threshold)
    logger.info("TP=%d  FP=%d  FN=%d", tp_total, fp_total, fn_total)
    logger.info("Precision=%.4f  Recall=%.4f  F1=%.4f", precision, recall, f1)
    if matched_box_ious:
        logger.info("Mean matched box-IoU : %.4f", np.mean(matched_box_ious))
    if matched_mask_ious:
        logger.info("Mean matched mask-IoU: %.4f", np.mean(matched_mask_ious))

    pbs = [r["pred_boxes"] for r in per_image]
    pss = [r["pred_scores"] for r in per_image]
    gbs = [r["gt_boxes"] for r in per_image]
    iou_range_coco = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    aps = compute_map(pbs, pss, gbs, iou_range_coco)
    ap50 = aps[0.5]; ap75 = aps[0.75]
    map_coco = float(np.nanmean(list(aps.values())))

    logger.info("-" * 60)
    logger.info("AP@0.50      = %.4f", ap50)
    logger.info("AP@0.75      = %.4f", ap75)
    logger.info("AP@[.5:.95]  = %.4f", map_coco)
    logger.info("-" * 60)

    figdir = args.output_dir
    os.makedirs(figdir, exist_ok=True)

    logger.info("Saving per-image overlays...")
    for rec in tqdm(per_image, desc="Viz", unit="wall"):
        img_np = rec["image_tensor"].permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        out_path = os.path.join(
            figdir, f"phase_1_eval_pred_{os.path.splitext(rec['filename'])[0]}.png"
        )
        visualize_wall(
            img_pil,
            rec["gt_boxes"], rec["gt_polygons"],
            rec["pred_boxes"], rec["pred_scores"], rec["pred_masks"],
            score_threshold=args.score_threshold,
            output_path=out_path,
            title=f"{rec['filename']}  GT={len(rec['gt_boxes'])}  Pred(>={args.score_threshold})={int((rec['pred_scores'] >= args.score_threshold).sum())}",
        )

    all_scores, all_tp = [], []
    for rec in per_image:
        matches, _ = greedy_match(
            rec["pred_boxes"], rec["pred_scores"], rec["gt_boxes"], args.iou_threshold
        )
        all_scores.extend(rec["pred_scores"].tolist())
        all_tp.extend([1 if m != -1 else 0 for m in matches])
    if all_scores:
        order = np.argsort(-np.asarray(all_scores))
        sorted_tp = np.asarray(all_tp)[order]
        tp_cum = np.cumsum(sorted_tp)
        fp_cum = np.cumsum(1 - sorted_tp)
        num_gt_total = sum(len(r["gt_boxes"]) for r in per_image)
        rec_curve = tp_cum / max(num_gt_total, 1)
        pre_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        plt.figure(figsize=(8, 6))
        plt.plot(rec_curve, pre_curve, linewidth=2)
        plt.xlim(0, 1); plt.ylim(0, 1.02)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR curve @ IoU={args.iou_threshold:.2f}   AP50={ap50:.3f}  AP75={ap75:.3f}")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figdir, "phase_1_eval_pr_curve.png"), dpi=180)
        plt.close()
 
    plt.figure(figsize=(9, 6))
    bins = np.linspace(0, 1, 21)
    plt.hist([score_tp, score_fp], bins=bins, stacked=True, label=[f"TP (n={len(score_tp)})", f"FP (n={len(score_fp)})"], color=["#2ca02c", "#d62728"], edgecolor="black")
    plt.axvline(args.score_threshold, color="black", linestyle="--", label=f"threshold={args.score_threshold}")
    plt.xlabel("Predicted score"); plt.ylabel("Count")
    plt.title("Prediction score distribution (at operating point)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_score_hist.png"), dpi=180)
    plt.close()

    if matched_box_ious:
        plt.figure(figsize=(9, 6))
        plt.hist(matched_box_ious, bins=np.linspace(args.iou_threshold, 1, 20), color="#1f77b4", edgecolor="black")
        plt.xlabel("Box IoU"); plt.ylabel("Count")
        plt.title(f"Box-IoU of matched predictions  (mean={np.mean(matched_box_ious):.3f})")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figdir, "phase_1_eval_box_iou_hist.png"), dpi=180)
        plt.close()

    if matched_mask_ious:
        plt.figure(figsize=(9, 6))
        plt.hist(matched_mask_ious, bins=np.linspace(0, 1, 20), color="#9467bd", edgecolor="black")
        plt.xlabel("Mask IoU"); plt.ylabel("Count")
        plt.title(f"Mask-IoU of matched predictions  (mean={np.mean(matched_mask_ious):.3f})")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figdir, "phase_1_eval_mask_iou_hist.png"), dpi=180)
        plt.close()

    fnames = [s["filename"] for s in per_img_stats]
    recalls = [s["recall"] for s in per_img_stats]
    plt.figure(figsize=(max(8, len(fnames) * 0.9), 6))
    bars = plt.bar(fnames, recalls, color="#17becf", edgecolor="black")
    plt.axhline(recall, color="black", linestyle="--", label=f"overall recall={recall:.3f}")
    plt.ylim(0, 1.05); plt.ylabel("Recall")
    plt.title(f"Per-image recall @ score>={args.score_threshold}, IoU>={args.iou_threshold}")
    plt.xticks(rotation=45, ha="right")
    for b, v in zip(bars, recalls):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_per_image_recall.png"), dpi=180)
    plt.close()

    gt_counts = [s["gt_count"] for s in per_img_stats]
    pred_counts = [s["pred_count_kept"] for s in per_img_stats]
    x = np.arange(len(fnames)); width = 0.38
    plt.figure(figsize=(max(8, len(fnames) * 0.9), 6))
    plt.bar(x - width / 2, gt_counts,   width, label="GT", color="#2ca02c", edgecolor="black")
    plt.bar(x + width / 2, pred_counts, width, label=f"Pred (>={args.score_threshold})", color="#d62728", edgecolor="black")
    plt.xticks(x, fnames, rotation=45, ha="right")
    plt.ylabel("Count"); plt.title("GT vs kept predictions per wall")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "phase_1_eval_gt_vs_pred_counts.png"), dpi=180)
    plt.close()

    metrics = {
        "operating_point": {
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "TP": int(tp_total), "FP": int(fp_total), "FN": int(fn_total),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "mean_matched_box_iou": float(np.mean(matched_box_ious))  if matched_box_ious  else None,
            "mean_matched_mask_iou": float(np.mean(matched_mask_ious)) if matched_mask_ious else None,
        },
        "mAP": {
            "AP@0.50": float(ap50),
            "AP@0.75": float(ap75),
            "AP@[.5:.95]": map_coco,
            "per_iou_threshold": {f"{t:.2f}": float(aps[t]) for t in iou_range_coco},
        },
        "per_image": per_img_stats,
        "config": {
            "checkpoint": args.checkpoint,
            "img_dir": args.img_dir,
            "ann_csv": args.ann_csv,
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
