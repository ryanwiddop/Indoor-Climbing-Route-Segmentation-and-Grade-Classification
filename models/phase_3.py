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
from matplotlib.path import Path
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

"""Convert raw grade prediction to the nearest valid grade index. (Rounding)"""
def to_grade(raw):
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

