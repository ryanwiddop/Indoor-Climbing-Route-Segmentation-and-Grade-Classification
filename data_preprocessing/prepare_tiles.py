"""
Tile full-wall images and emit train/val tile-level COCO files.

Input:  full-res image dir + walls_clean.json (COCO, full-wall polygons).
Output: out_dir/{tiles/, visualisations/, tiles_train.json, tiles_val.json}

The split is done at the WALL level (not the tile level) using --seed, so all
tiles from a given source wall go entirely to one split.

Usage:
    python prepare_tiles.py <img_dir> <coco_json> <out_dir> [options]
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

from PIL import Image, ImageDraw, ImageOps


def iter_tiles(H, W, tile, overlap):
    stride = max(1, int(round(tile * (1 - overlap))))

    def axis_positions(length):
        positions = list(range(0, max(0, length - tile) + 1, stride))
        if positions and positions[-1] + tile < length:
            positions.append(length - tile)
        return sorted(set(positions))

    for y0 in axis_positions(H):
        for x0 in axis_positions(W):
            yield (x0, y0, x0 + tile, y0 + tile)


def polygon_bbox_overlap_fraction(xs, ys, tx0, ty0, tx1, ty1):
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    inter_xmin = max(x_min, tx0)
    inter_ymin = max(y_min, ty0)
    inter_xmax = min(x_max, tx1)
    inter_ymax = min(y_max, ty1)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    hold_area = (x_max - x_min) * (y_max - y_min)
    if hold_area == 0:
        return 0.0
    return inter_area / hold_area


def translate(xs, ys, ox, oy):
    return [x - ox for x in xs], [y - oy for y in ys]


def polygon_area(xs, ys):
    n = len(xs)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


def coco_bbox(xs, ys):
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def flat_seg(xs, ys):
    coords = []
    for x, y in zip(xs, ys):
        coords.extend([float(x), float(y)])
    return [coords]


def extract_polygon(seg):
    if not seg or not isinstance(seg, list):
        return [], []
    first = seg[0]
    if not first:
        return [], []
    return list(first[::2]), list(first[1::2])


def draw_tile_annotations(tile_img, holds, cat_id_to_name):
    vis = tile_img.copy()
    draw = ImageDraw.Draw(vis)
    for hold in holds:
        polygon = list(zip(hold["xs"], hold["ys"]))
        if len(polygon) >= 2:
            draw.polygon(polygon, outline="yellow", width=3)
        x0 = min(hold["xs"])
        y0 = min(hold["ys"])
        label = cat_id_to_name.get(hold["category_id"], str(hold["category_id"]))
        draw.text((x0, y0), label, fill="yellow")
    return vis


def split_walls(image_names, val_frac, seed):
    rng = random.Random(seed)
    shuffled = sorted(image_names)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(val_frac * len(shuffled))))
    val = set(shuffled[:n_val])
    train = set(shuffled[n_val:])
    return train, val


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("img_dir")
    p.add_argument("coco_json")
    p.add_argument("out_dir")
    p.add_argument("--tile-size", type=int, default=800)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--min-overlap-frac", type=float, default=0.35,
                   help="Min fraction of hold bbox that must overlap tile")
    p.add_argument("--vis-per-image", type=int, default=3,
                   help="Positive-tile visualisations saved per source wall")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    tiles_dir = os.path.join(args.out_dir, "tiles")
    vis_dir = os.path.join(args.out_dir, "visualisations")
    for d in [tiles_dir, vis_dir]:
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

    next_image_id = 1
    next_ann_id = 1
    train_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    val_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    stats = {"tiles": 0, "pos": 0, "neg": 0, "holds": 0, "skipped": 0}

    for wall in coco["images"]:
        img_name = wall["file_name"]
        img_path = os.path.join(args.img_dir, img_name)
        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} not found. Skipping.")
            stats["skipped"] += 1
            continue

        image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        W, H = image.size
        img_stem = os.path.splitext(img_name)[0]
        target = train_coco if img_name in train_walls else val_coco

        holds = []
        for ann in anns_by_image[wall["id"]]:
            xs, ys = extract_polygon(ann["segmentation"])
            if not xs or len(xs) != len(ys):
                continue
            if max(xs) <= min(xs) or max(ys) <= min(ys):
                continue
            holds.append({
                "xs": xs,
                "ys": ys,
                "category_id": ann["category_id"],
                "attributes": ann.get("attributes", {}),
            })

        tile_records = []
        for (tx0, ty0, tx1, ty1) in iter_tiles(H, W, args.tile_size, args.overlap):
            tile_name = f"{img_stem}_t{tx0}_{ty0}.jpg"
            tile_path = os.path.join(tiles_dir, tile_name)
            tile_crop = image.crop((tx0, ty0, tx1, ty1))
            tile_crop.save(tile_path, "JPEG", quality=95)
            stats["tiles"] += 1

            tile_w, tile_h = tile_crop.size
            tile_image_id = next_image_id
            next_image_id += 1
            target["images"].append({
                "id": tile_image_id,
                "file_name": tile_name,
                "width": tile_w,
                "height": tile_h,
                "source_image": img_name,
                "tile_x0": tx0,
                "tile_y0": ty0,
                "tile_x1": tx1,
                "tile_y1": ty1,
            })

            qualifying = []
            for hold in holds:
                frac = polygon_bbox_overlap_fraction(hold["xs"], hold["ys"], tx0, ty0, tx1, ty1)
                if frac < args.min_overlap_frac:
                    continue
                lxs, lys = translate(hold["xs"], hold["ys"], tx0, ty0)
                lxs = [max(0, min(tile_w, x)) for x in lxs]
                lys = [max(0, min(tile_h, y)) for y in lys]
                if max(lxs) - min(lxs) <= 0 or max(lys) - min(lys) <= 0:
                    continue
                qualifying.append({**hold, "xs": lxs, "ys": lys})

            if qualifying:
                stats["pos"] += 1
                stats["holds"] += len(qualifying)
                for hold in qualifying:
                    target["annotations"].append({
                        "id": next_ann_id,
                        "image_id": tile_image_id,
                        "category_id": hold["category_id"],
                        "segmentation": flat_seg(hold["xs"], hold["ys"]),
                        "area": round(polygon_area(hold["xs"], hold["ys"]), 2),
                        "bbox": coco_bbox(hold["xs"], hold["ys"]),
                        "iscrowd": 0,
                        "attributes": hold["attributes"],
                    })
                    next_ann_id += 1
            else:
                stats["neg"] += 1

            tile_records.append((tile_name, tile_crop, qualifying))

        pos_records = [(n, c, q) for n, c, q in tile_records if q]
        sample_n = min(args.vis_per_image, len(pos_records))
        for tile_name, tile_crop, qualifying in rng.sample(pos_records, sample_n):
            vis = draw_tile_annotations(tile_crop, qualifying, cat_id_to_name)
            vis_name = f"{os.path.splitext(tile_name)[0]}_vis.jpg"
            vis.save(os.path.join(vis_dir, vis_name), "JPEG", quality=95)

        split = "train" if img_name in train_walls else "val"
        print(f"   {img_name} ({split}): {len(holds)} holds -> "
              f"{stats['tiles']} tiles, {stats['pos']} pos / {stats['neg']} neg")

    train_path = os.path.join(args.out_dir, "tiles_train.json")
    val_path = os.path.join(args.out_dir, "tiles_val.json")
    with open(train_path, "w") as f:
        json.dump(train_coco, f)
    with open(val_path, "w") as f:
        json.dump(val_coco, f)

    print()
    print("=" * 60)
    print(f"Tile size           : {args.tile_size}px  overlap={args.overlap}")
    print(f"Min hold overlap    : {args.min_overlap_frac:.0%}")
    print(f"Total tiles         : {stats['tiles']}")
    print(f"  Positive          : {stats['pos']}")
    print(f"  Negative          : {stats['neg']}")
    if stats["pos"] > 0:
        print(f"  Neg:Pos ratio     : {stats['neg'] / stats['pos']:.2f}")
        print(f"Hold instances      : {stats['holds']}  ({stats['holds'] / stats['pos']:.1f} per pos tile)")
    print(f"Images skipped      : {stats['skipped']}")
    print(f"Train COCO          : {train_path}  ({len(train_coco['images'])} tiles, {len(train_coco['annotations'])} anns)")
    print(f"Val COCO            : {val_path}  ({len(val_coco['images'])} tiles, {len(val_coco['annotations'])} anns)")
    print(f"Tiles dir           : {tiles_dir}")
    print(f"Visualizations      : {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
