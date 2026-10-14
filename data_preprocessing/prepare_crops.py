"""
Crop every annotated hold to its own standardized image and emit train/val
crop-level COCO files.

Input:  full-res image dir + walls_clean.json (COCO, full-wall polygons).
Output: out_dir/{crops/, visualisations/, crops_train.json, crops_val.json}

Each hold bbox is expanded to a square about its centre, padded by --margin,
then letterboxed to --size x --size so hold type cannot be read off absolute
size. Polygons are rewritten into resized crop-local coordinates. The original
bbox geometry is preserved on each record (source_w/source_h/source_aspect)
because the square crop no longer carries it.

The split is done at the WALL level using --seed, via the same split_walls as
prepare_tiles.py and fed the same full image list, so crop/tile/tabular splits
agree.

Usage:
    python prepare_crops.py <img_dir> <coco_json> <out_dir> [options]
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageOps

from prepare_tiles import (
    coco_bbox,
    extract_polygon,
    flat_seg,
    polygon_area,
    split_walls,
)

SIZE_BINS = [0, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]


def square_window(xs, ys, W, H, margin):
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    side = max(x_max - x_min, y_max - y_min) * (1.0 + margin)
    side = min(side, float(min(W, H)))

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    x0 = cx - side / 2.0
    y0 = cy - side / 2.0

    x0 = min(max(0.0, x0), max(0.0, W - side))
    y0 = min(max(0.0, y0), max(0.0, H - side))

    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = min(W, x0 + int(round(side)))
    y1 = min(H, y0 + int(round(side)))
    return x0, y0, x1, y1


def letterbox(crop, size):
    cw, ch = crop.size
    scale = size / max(cw, ch)
    new_w = max(1, min(size, int(round(cw * scale))))
    new_h = max(1, min(size, int(round(ch * scale))))

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(crop.resize((new_w, new_h), Image.BILINEAR), (left, top))
    return canvas, scale, left, top


def to_crop_space(xs, ys, x0, y0, scale, left, top, size):
    lxs = [max(0.0, min(size, (x - x0) * scale + left)) for x in xs]
    lys = [max(0.0, min(size, (y - y0) * scale + top)) for y in ys]
    return lxs, lys


def draw_crop_annotation(crop_img, xs, ys, label):
    vis = crop_img.copy()
    draw = ImageDraw.Draw(vis)
    polygon = list(zip(xs, ys))
    if len(polygon) >= 2:
        draw.polygon(polygon, outline="yellow", width=2)
    draw.text((2, 2), label, fill="yellow")
    return vis


def size_histogram(short_sides):
    counts = Counter()
    for s in short_sides:
        bucket = SIZE_BINS[0]
        for edge in SIZE_BINS:
            if s >= edge:
                bucket = edge
        counts[bucket] += 1
    return counts


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("img_dir")
    p.add_argument("coco_json")
    p.add_argument("out_dir")
    p.add_argument("--size", type=int, default=224,
                   help="Output crop edge length in px")
    p.add_argument("--margin", type=float, default=0.15,
                   help="Fractional increase of the square side around the hold bbox")
    p.add_argument("--min-size-warn", type=int, default=32,
                   help="Report holds whose source bbox shorter side is below this")
    p.add_argument("--vis-per-category", type=int, default=6,
                   help="Crop visualisations saved per hold category")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    crops_dir = os.path.join(args.out_dir, "crops")
    vis_dir = os.path.join(args.out_dir, "visualisations")
    for d in [crops_dir, vis_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    with open(args.coco_json) as f:
        coco = json.load(f)
    categories = coco["categories"]
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    wall_names = [im["file_name"] for im in coco["images"]]
    train_walls, val_walls = split_walls(wall_names, args.val_frac, args.seed)
    print(f"Wall-level split: {len(train_walls)} train, {len(val_walls)} val")
    print(f"  Val walls: {sorted(val_walls)}")

    def empty_coco():
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": categories,
            "images": [],
            "annotations": [],
        }

    train_coco = empty_coco()
    val_coco = empty_coco()

    next_image_id = 1
    stats = {"crops": 0, "degenerate": 0, "skipped": 0, "unannotated": 0, "tiny": 0}
    per_cat = Counter()
    per_cat_split = defaultdict(Counter)
    short_sides = []
    vis_pool = defaultdict(list)

    for wall in coco["images"]:
        img_name = wall["file_name"]
        holds = anns_by_image[wall["id"]]

        if not holds:
            print(f"   {img_name}: 0 annotated holds. Skipping.")
            stats["unannotated"] += 1
            continue

        img_path = os.path.join(args.img_dir, img_name)
        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} not found. Skipping.")
            stats["skipped"] += 1
            continue

        image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        W, H = image.size
        img_stem = os.path.splitext(img_name)[0]
        split = "train" if img_name in train_walls else "val"
        target = train_coco if split == "train" else val_coco

        n_wall = 0
        for ann in holds:
            xs, ys = extract_polygon(ann["segmentation"])
            if not xs or len(xs) != len(ys) or len(xs) < 3:
                stats["degenerate"] += 1
                continue
            xs = [max(0.0, min(float(W), x)) for x in xs]
            ys = [max(0.0, min(float(H), y)) for y in ys]
            src_w = max(xs) - min(xs)
            src_h = max(ys) - min(ys)
            if src_w <= 0 or src_h <= 0:
                stats["degenerate"] += 1
                continue

            x0, y0, x1, y1 = square_window(xs, ys, W, H, args.margin)
            if x1 <= x0 or y1 <= y0:
                stats["degenerate"] += 1
                continue

            crop, scale, left, top = letterbox(image.crop((x0, y0, x1, y1)), args.size)
            lxs, lys = to_crop_space(xs, ys, x0, y0, scale, left, top, args.size)
            if max(lxs) - min(lxs) <= 0 or max(lys) - min(lys) <= 0:
                stats["degenerate"] += 1
                continue

            cat_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            crop_name = f"{img_stem}_{ann['id']}.jpg"
            crop.save(os.path.join(crops_dir, crop_name), "JPEG", quality=95)

            short = min(src_w, src_h)
            short_sides.append(short)
            if short < args.min_size_warn:
                stats["tiny"] += 1

            crop_image_id = next_image_id
            next_image_id += 1
            target["images"].append({
                "id": crop_image_id,
                "file_name": crop_name,
                "width": args.size,
                "height": args.size,
                "source_image": img_name,
                "source_ann_id": ann["id"],
                "source_w": round(src_w, 2),
                "source_h": round(src_h, 2),
                "source_aspect": round(src_h / src_w, 4),
                "crop_x0": x0,
                "crop_y0": y0,
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_scale": round(scale, 6),
                "pad_left": left,
                "pad_top": top,
            })
            target["annotations"].append({
                "id": crop_image_id,
                "image_id": crop_image_id,
                "category_id": ann["category_id"],
                "segmentation": flat_seg(lxs, lys),
                "area": round(polygon_area(lxs, lys), 2),
                "bbox": coco_bbox(lxs, lys),
                "iscrowd": 0,
                "attributes": ann.get("attributes", {}),
                "source_area": ann.get("area"),
            })

            per_cat[cat_name] += 1
            per_cat_split[cat_name][split] += 1
            stats["crops"] += 1
            n_wall += 1
            vis_pool[cat_name].append((crop_name, crop, lxs, lys))

        print(f"   {img_name} ({split}): {len(holds)} holds -> {n_wall} crops")

    for cat_name, pool in vis_pool.items():
        for crop_name, crop, lxs, lys in rng.sample(pool, min(args.vis_per_category, len(pool))):
            vis = draw_crop_annotation(crop, lxs, lys, cat_name)
            vis.save(os.path.join(vis_dir, f"{cat_name}_{os.path.splitext(crop_name)[0]}_vis.jpg"),
                     "JPEG", quality=95)

    train_path = os.path.join(args.out_dir, "crops_train.json")
    val_path = os.path.join(args.out_dir, "crops_val.json")
    with open(train_path, "w") as f:
        json.dump(train_coco, f)
    with open(val_path, "w") as f:
        json.dump(val_coco, f)

    print()
    print("=" * 60)
    print(f"Crop size           : {args.size}px  margin={args.margin:.0%}")
    print(f"Total crops         : {stats['crops']}")
    print(f"Degenerate polygons : {stats['degenerate']}")
    print(f"Images skipped      : {stats['skipped']} missing, {stats['unannotated']} unannotated")
    print()
    print(f"{'category':<10} {'total':>6} {'train':>6} {'val':>6}")
    for cat in categories:
        name = cat["name"]
        print(f"{name:<10} {per_cat[name]:>6} {per_cat_split[name]['train']:>6} "
              f"{per_cat_split[name]['val']:>6}")
    print()
    print("Source bbox shorter side (px):")
    hist = size_histogram(short_sides)
    for i, edge in enumerate(SIZE_BINS):
        n = hist.get(edge, 0)
        if n == 0:
            continue
        hi = SIZE_BINS[i + 1] if i + 1 < len(SIZE_BINS) else None
        label = f"{edge}-{hi}" if hi else f"{edge}+"
        bar = "#" * max(1, round(40 * n / max(1, len(short_sides))))
        print(f"  {label:>8} {n:>5}  {bar}")
    if short_sides:
        print(f"  below {args.min_size_warn}px : {stats['tiny']} crops "
              f"({stats['tiny'] / len(short_sides):.1%}) upsampled to {args.size}px")
    print()
    print(f"Train COCO          : {train_path}  ({len(train_coco['images'])} crops)")
    print(f"Val COCO            : {val_path}  ({len(val_coco['images'])} crops)")
    print(f"Crops dir           : {crops_dir}")
    print(f"Visualizations      : {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
