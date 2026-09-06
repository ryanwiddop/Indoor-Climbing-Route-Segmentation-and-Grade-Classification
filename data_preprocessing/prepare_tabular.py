"""
Prepare tabular data from full-wall images for phase 2 training. So route ids must all be merged since they are currently per image. Then

Input:  walls_clean.json (COCO, full-wall polygons).
Output: {out_dir}/{tabular_data.csv}

Usage:
    python prepare_tabular.py <coco_json> <out_dir> [options]
"""

import argparse
import json
import random
import numpy as np
import csv
import os

from collections import defaultdict
from dataclasses import dataclass
from scipy.spatial import QhullError, distance, ConvexHull

from prepare_tiles import split_walls


IGNORE_CATEGORIES = {"volume", "hold"}
HAND_CATEGORIES = {"jug", "crimp", "pocket", "sloper", "pinch", "edge"}
FOOT_CATEGORIES = {"foot"}
FEATURE_CATEGORIES = HAND_CATEGORIES | FOOT_CATEGORIES

# Grid config for spliting feature locations (TUNE THIS LARGER FOR GRANULARITY if underperfoming)
GRID_ROWS = 3
GRID_COLS = 3

VOLUME_MARGIN_HOLD_UNITS = 2.0 # margin around bbox of route to add volumes not in bbox
DROP_GRADES = {"INC"} # drop routes with these grades

@dataclass
class CocoData:
    raw: dict
    cat_id_to_name: dict
    img_id_to_anns: dict
    img_by_id: dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("coco_json", type=str, help="COCO json file with full-wall polygons")
    p.add_argument("out_dir", type=str, help="output directory for tabular data")
    p.add_argument("--min-holds", type=int, default=3, help="minimum holds per route to include")
    p.add_argument("--include-volumes", action="store_true", help="include volumes in the dataset")
    p.add_argument("--val-frac", type=float, default=0.2, help="fraction of walls to use for validation")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducibility") # same seed as prepare tiles
    return p.parse_args()


def load_coco(coco_json):
    with open(coco_json, "r") as f:
        coco = json.load(f)
        
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_id_to_anns[ann["image_id"]].append(ann)
    img_by_id = {img["id"]: img for img in coco["images"]}
    return CocoData(raw=coco, cat_id_to_name=cat_id_to_name, img_id_to_anns=img_id_to_anns, img_by_id=img_by_id)


def hold_area(ann):
    return ann["area"]


def hold_diag(ann):
    return ann["area"] ** 0.5


def hold_centroid(hold):
    """[x, y, w, h] -> (cx, cy)"""
    x, y, w, h = hold["bbox"]
    return (x + w / 2, y + h / 2)


def image_scale(anns, coco: CocoData):
    """median sqrt(area) over non-volume, non-hold annotations in this image."""
    if not anns:
        return None
    
    scales = []
    for ann in anns:
        if coco.cat_id_to_name[ann["category_id"]] not in IGNORE_CATEGORIES:
            scales.append(hold_diag(ann))
            
    if not scales:
        return None
    return sorted(scales)[len(scales) // 2] if scales else None


def group_routes(coco: CocoData):
    """Group annotations by (image_id, route_id) and filter out ignored categories. Returns a dict mapping (image_id, route_id) -> list of annotations."""
    routes = defaultdict(list)
    for ann in coco.raw["annotations"]:
        image_id = ann["image_id"]
        route_id = ann.get("attributes", {}).get("route_id")
        if route_id is None or coco.cat_id_to_name[ann["category_id"]] in IGNORE_CATEGORIES:
            continue
        routes[(image_id, route_id)].append(ann)
    return routes


def pairwise_distance(points):
    """Compute pairwise distances between points. Points is a list of (x, y) tuples."""
    points = np.array(points)
    return distance.cdist(points, points, metric='euclidean')


def nearest_neighbor_distances(points):
    """Compute the nearest neighbor distance for each point in points. Points is a list of (x, y) tuples."""
    dist_matrix = pairwise_distance(points)
    np.fill_diagonal(dist_matrix, np.inf)  # ignore self distance
    return np.min(dist_matrix, axis=1)


def convex_hull_area(points):
    """Compute the area of the convex hull of a set of points. Points is a list of (x, y) tuples."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume 
    except QhullError:
        return 0.0


def bbox_of_points(points):
    """Compute the bounding box of a set of points. Points is a list of (x, y) tuples."""
    xs, ys = zip(*points)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0, y0, x1, y1)


def bbox_expanded(bbox, margin):
    """Expand a bounding box by a margin. bbox is (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = bbox
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def bbox_intersects(bbox1, bbox2):
    """Check if two bounding boxes intersect. Each bbox is (x0, y0, x1, y1)."""
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    return not (x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1)


###################### FEATURE EXTRACTION FUNCTIONS ######################


def features_counts(coco: CocoData, route_holds):
    """
    n_jug, n_crimp, n_pocket, n_sloper, n_pinch, n_edge, n_foot,
    frac_jug, frac_crimp, frac_pocket, frac_sloper, frac_pinch, frac_edge, frac_foot
    """
    counts = defaultdict(int)
    for ann in route_holds:
        cat_name = coco.cat_id_to_name[ann["category_id"]]
        if cat_name in FEATURE_CATEGORIES:
            counts[cat_name] += 1
    
    total = sum(counts[cat] for cat in FEATURE_CATEGORIES)
    if total == 0:
        return {f"n_{cat}": 0 for cat in FEATURE_CATEGORIES} | {f"frac_{cat}": 0.0 for cat in FEATURE_CATEGORIES}
    else:
        return {f"n_{cat}": counts[cat] for cat in FEATURE_CATEGORIES} | {f"frac_{cat}": counts[cat] / total for cat in FEATURE_CATEGORIES}
    
    
def features_size_per_category(coco: CocoData, route_holds, scale):
    """
    mean_area_jug, mean_area_crimp, mean_area_pocket, mean_area_sloper, mean_area_pinch, mean_area_edge, mean_area_foot,
    max_area_jug, max_area_crimp, max_area_pocket, max_area_sloper, max_area_pinch, max_area_edge, max_area_foot,
    min_area_jug, min_area_crimp, min_area_pocket, min_area_sloper, min_area_pinch, min_area_edge, min_area_foot,
    std_area_jug, std_area_crimp, std_area_pocket, std_area_sloper, std_area_pinch, std_area_edge, std_area_foot
    """
    areas_by_cat = defaultdict(list)
    for ann in route_holds:
        cat_name = coco.cat_id_to_name[ann["category_id"]]
        if cat_name in FEATURE_CATEGORIES:
            areas_by_cat[cat_name].append(hold_area(ann))
    
    normalized_areas_by_cat = {}
    for cat in FEATURE_CATEGORIES:
        areas = areas_by_cat[cat]
        if areas:
            normalized_areas_by_cat[cat] = [area / (scale ** 2) for area in areas]
        else:
            normalized_areas_by_cat[cat] = []

    features = {}
    for cat in FEATURE_CATEGORIES:
        areas = normalized_areas_by_cat[cat]
        if areas:
            features[f"mean_area_{cat}"] = np.mean(areas)
            features[f"max_area_{cat}"] = np.max(areas)
            features[f"min_area_{cat}"] = np.min(areas)
            features[f"std_area_{cat}"] = np.std(areas)
        else:
            features[f"mean_area_{cat}"] = float("nan")
            features[f"max_area_{cat}"] = float("nan")
            features[f"min_area_{cat}"] = float("nan")
            features[f"std_area_{cat}"] = float("nan")
    return features


def features_inter_category_distances(coco: CocoData, route_holds, scale):
    """
    mean_hand_to_hand_distance, mean_hand_to_foot_distance, mean_foot_to_foot_distance,
    max_hand_to_hand_distance, max_hand_to_foot_distance, max_foot_to_foot_distance,
    min_hand_to_hand_distance, min_hand_to_foot_distance, min_foot_to_foot_distance,
    std_hand_to_hand_distance, std_hand_to_foot_distance, std_foot_to_foot_distance
    """
    hand_points = []
    foot_points = []
    for ann in route_holds:
        cat_name = coco.cat_id_to_name[ann["category_id"]]
        if cat_name in HAND_CATEGORIES:
            hand_points.append(hold_centroid(ann))
        elif cat_name in FOOT_CATEGORIES:
            foot_points.append(hold_centroid(ann))
    
    distances = {
        "hand_to_hand": [],
        "hand_to_foot": [],
        "foot_to_foot": []
    }
    
    if len(hand_points) > 1:
        dists = nearest_neighbor_distances(hand_points)
        distances["hand_to_hand"].extend(dists)
    
    if len(foot_points) > 1:
        dists = nearest_neighbor_distances(foot_points)
        distances["foot_to_foot"].extend(dists)
    
    if hand_points and foot_points:
        for h in hand_points:
            for f in foot_points:
                distances["hand_to_foot"].append(distance.euclidean(h, f))
    
    features = {}
    for key, dists in distances.items():
        if dists:
            normalized_dists = [d / scale for d in dists]
            features[f"mean_{key}_distance"] = np.mean(normalized_dists)
            features[f"max_{key}_distance"] = np.max(normalized_dists)
            features[f"min_{key}_distance"] = np.min(normalized_dists)
            features[f"std_{key}_distance"] = np.std(normalized_dists)
        else:
            features[f"mean_{key}_distance"] = float("nan")
            features[f"max_{key}_distance"] = float("nan")
            features[f"min_{key}_distance"] = float("nan")
            features[f"std_{key}_distance"] = float("nan")
    
    return features


def features_grid(coco: CocoData, route_holds, route_bbox):
    """
    For each grid cell, count the number of holds in that cell. Return a dict with keys like "grid_0_0", "grid_0_1", ..., "grid_2_2".
    """
    x0, y0, x1, y1 = route_bbox
    grid_width = (x1 - x0) / GRID_COLS if (x1 - x0) > 0 else 1.0
    grid_height = (y1 - y0) / GRID_ROWS if (y1 - y0) > 0 else 1.0
    
    grid_hand_counts = {f"grid_hands_r{r}_c{c}": 0 for r in range(GRID_ROWS) for c in range(GRID_COLS)}
    grid_foot_counts = {f"grid_feet_r{r}_c{c}": 0 for r in range(GRID_ROWS) for c in range(GRID_COLS)}
    
    for ann in route_holds:
        cx, cy = hold_centroid(ann)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            col = min(int((cx - x0) / grid_width), GRID_COLS - 1)
            row = min(int((cy - y0) / grid_height), GRID_ROWS - 1)
            cat_name = coco.cat_id_to_name[ann["category_id"]]
            if cat_name in HAND_CATEGORIES:
                grid_hand_counts[f"grid_hands_r{row}_c{col}"] += 1
            elif cat_name in FOOT_CATEGORIES:
                grid_foot_counts[f"grid_feet_r{row}_c{col}"] += 1
    
    return {**grid_hand_counts, **grid_foot_counts}


def features_extent(route_holds, scale):
    """span_x, span_y, aspect, bbox_area, convex_hull_area, hold_density"""
    if not route_holds:
        return {
            "span_x": float("nan"),
            "span_y": float("nan"),
            "aspect": float("nan"),
            "bbox_area": float("nan"),
            "convex_hull_area": float("nan"),
            "hold_density": float("nan")
        }
    
    points = [hold_centroid(ann) for ann in route_holds]
    xs, ys = zip(*points)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    
    span_x = (x1 - x0) / scale
    span_y = (y1 - y0) / scale
    aspect = span_x / span_y if span_y != 0 else float("nan")
    bbox_area = (span_x * span_y)
    convex_hull_area_value = convex_hull_area(points) / (scale ** 2)
    hold_density = len(route_holds) / bbox_area if bbox_area > 0 else float("nan")
    
    return {
        "span_x": span_x,
        "span_y": span_y,
        "aspect": aspect,
        "bbox_area": bbox_area,
        "convex_hull_area": convex_hull_area_value,
        "hold_density": hold_density
    }
    
    
def features_distances(route_holds, scale):
    """
    cg_mean, cg_std, cg_min, cg_max, cg_median,
    nn_mean, nn_std, nn_min, nn_max, nn_median
    """
    if not route_holds:
        return {
            "cg_mean": float("nan"),
            "cg_std": float("nan"),
            "cg_min": float("nan"),
            "cg_max": float("nan"),
            "cg_median": float("nan"),
            "nn_mean": float("nan"),
            "nn_std": float("nan"),
            "nn_min": float("nan"),
            "nn_max": float("nan"),
            "nn_median": float("nan")
        }
    
    points = [hold_centroid(ann) for ann in route_holds]
    cg_x = np.mean([p[0] for p in points])
    cg_y = np.mean([p[1] for p in points])
    cg_distances = [distance.euclidean((cg_x, cg_y), p) / scale for p in points]
    
    nn_distances = nearest_neighbor_distances(points)
    nn_distances_normalized = [d / scale for d in nn_distances]
    
    features = {
        "cg_mean": np.mean(cg_distances),
        "cg_std": np.std(cg_distances),
        "cg_min": np.min(cg_distances),
        "cg_max": np.max(cg_distances),
        "cg_median": np.median(cg_distances),
        "nn_mean": np.mean(nn_distances_normalized),
        "nn_std": np.std(nn_distances_normalized),
        "nn_min": np.min(nn_distances_normalized),
        "nn_max": np.max(nn_distances_normalized),
        "nn_median": np.median(nn_distances_normalized)
    }
    
    return features


def features_path(route_holds, scale):
    """
    path_length, path_straightness
    """
    if not route_holds:
        return {
            "path_length": float("nan"),
            "path_straightness": float("nan"),
        }
    
    points = [hold_centroid(ann) for ann in route_holds]
    if len(points) < 2:
        return {
            "path_length": 0.0,
            "path_straightness": float("nan"),
        }
    
    points_sorted = sorted(points, key=lambda p: p[1])
    
    path_length = sum(distance.euclidean(points_sorted[i], points_sorted[i + 1]) for i in range(len(points_sorted) - 1)) / scale
    
    straight_line_distance = distance.euclidean(points_sorted[0], points_sorted[-1]) / scale
    path_straightness = straight_line_distance / path_length if path_length > 0 else float("nan")
        
    return {
        "path_length": path_length,
        "path_straightness": path_straightness
    }
    
    
def features_volumes(coco: CocoData, route_bbox, scale, volumes_for_image):
    """
    n_volumes_nearby, frac_route_bbox_covered_by_volumes
    """
    expanded_bbox = bbox_expanded(route_bbox, VOLUME_MARGIN_HOLD_UNITS * scale)
    n_volumes_nearby = 0
    volume_areas = []
    for ann in volumes_for_image:
        cat_name = coco.cat_id_to_name[ann["category_id"]]
        if cat_name == "volume":
            volume_bbox = ann["bbox"]
            volume_bbox_xyxy = (volume_bbox[0], volume_bbox[1], volume_bbox[0] + volume_bbox[2], volume_bbox[1] + volume_bbox[3])
            if bbox_intersects(expanded_bbox, volume_bbox_xyxy):
                n_volumes_nearby += 1
                volume_areas.append(hold_area(ann) / (scale ** 2))

    x0, y0, x1, y1 = route_bbox
    route_bbox_area_normalized = (x1 - x0) * (y1 - y0) / (scale ** 2)
    if route_bbox_area_normalized > 0:
        frac = min(sum(volume_areas) / route_bbox_area_normalized, 1.0)
    else:
        frac = float("nan")

    return {
        "n_volumes_nearby": n_volumes_nearby,
        "frac_route_bbox_covered_by_volumes": frac
    }
    
    
def build_row(image_rec, route_id_str, route_holds, coco: CocoData, scale, volumes_for_image):
    """
    Build a row of features for a given route.
    """
    row = {
        "image_id": image_rec["id"],
        "route_id": route_id_str,
        "n_holds": len(route_holds),
        "route_grade": route_holds[0].get("attributes", {}).get("route_grade", "INC") if route_holds else "INC"
    }
    
    points = [hold_centroid(ann) for ann in route_holds]
    if points:
        x0, y0, x1, y1 = bbox_of_points(points)
        route_bbox = (x0, y0, x1, y1)
    else:
        route_bbox = (0, 0, 0, 0)
    
    row.update(features_counts(coco, route_holds))
    row.update(features_size_per_category(coco, route_holds, scale))
    row.update(features_inter_category_distances(coco, route_holds, scale))
    row.update(features_grid(coco, route_holds, route_bbox))
    row.update(features_extent(route_holds, scale))
    row.update(features_distances(route_holds, scale))
    row.update(features_path(route_holds, scale))
    row.update(features_volumes(coco, route_bbox, scale, volumes_for_image))
    
    return row


def write_csv(rows, out_path):
    if not rows:
        print(f"No rows to write to {out_path}.")
        return
    
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


############################ MAIN FUNCTION ###########################
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    coco = load_coco(args.coco_json)
    anns = coco.raw["annotations"]
    
    scales = {img_id: image_scale(coco.img_id_to_anns[img_id], coco) for img_id in coco.img_by_id.keys()}
    
    volumes_by_image = defaultdict(list)
    if args.include_volumes:
        for ann in anns:
            cat_name = coco.cat_id_to_name[ann["category_id"]]
            if cat_name == "volume":
                volumes_by_image[ann["image_id"]].append(ann)
                
    routes = group_routes(coco)
    
    wall_names = [im["file_name"] for im in coco.raw["images"]]
    train_walls, val_walls = split_walls(wall_names, args.val_frac, args.seed)
    print(f"Wall-level split: {len(train_walls)} train, {len(val_walls)} val")
    
    train_rows = []
    val_rows = []
    stats = {"total": 0, "dropped_grade": 0, "dropped_size": 0, "kept": 0}
    
    for (image_id, route_id), route_holds in routes.items():
        stats["total"] += 1
        image_rec = coco.img_by_id[image_id]
        route_id_str = str(route_id)
        
        if route_holds[0].get("attributes", {}).get("route_grade", "INC") in DROP_GRADES:
            stats["dropped_grade"] += 1
            continue
        
        if len(route_holds) < args.min_holds:
            stats["dropped_size"] += 1
            continue
        
        row = build_row(image_rec, route_id_str, route_holds, coco, scales[image_id], volumes_by_image[image_id])
        
        wall_name = image_rec["file_name"]
        if wall_name in train_walls:
            train_rows.append(row)
        elif wall_name in val_walls:
            val_rows.append(row)
        
        stats["kept"] += 1
        
    print(f"Stats: {stats}")
    print(f"Features count: {len(train_rows[0]) if train_rows else 0}")
    write_csv(train_rows, os.path.join(args.out_dir, "train_tabular_data.csv"))
    write_csv(val_rows, os.path.join(args.out_dir, "val_tabular_data.csv"))


if __name__ == "__main__":
    main()