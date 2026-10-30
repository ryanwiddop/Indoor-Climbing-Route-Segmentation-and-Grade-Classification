import os
import logging
import cv2
from collections import defaultdict
import time
import json

import numpy as np
import pandas as pd

WHOLE_IMG_PATH = "/home/public/rwiddop/images/"
ANNOTATIONS_PATH = "/home/public/rwiddop/annotations.json"
    
    
def load_annotations(annotations_path, image_files):
    with open(annotations_path, "r") as f:
        data = json.load(f)
    name_to_id = {img["file_name"]: img["id"] for img in data["images"]}
    wanted = {name_to_id[f]: f for f in image_files if f in name_to_id}
    image_to_annotations = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] in wanted:
            image_to_annotations[wanted[ann["image_id"]]].append(ann)
    return image_to_annotations


def extract_clab_from_ann(ann, lab_image):
    """
    "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "segmentation": [[2624.0, 1212.0, 2629.0, 1211.0, 2638.0, 1204.0, 2643.0, 1193.0, 2645.0, 1185.0, 2642.0, 1180.0, 2634.0, 1174.0, 2626.0, 1172.0, 2619.0, 1172.0, 2611.0, 1176.0, 2606.0, 1179.0, 2592.0, 1183.0, 2590.0, 1183.0, 2586.0, 1184.0, 2584.0, 1189.0, 2586.0, 1196.0, 2609.0, 1210.0]], "area": 1675.0, "bbox": [2584.0, 1172.0, 61.0, 40.0], "iscrowd": 0, "attributes": {"route_id": 1, "route_grade": "V0-"}},
    """
    bx, by, bw, bh = (int(v) for v in ann["bbox"])
    x, y = max(bx, 0), max(by, 0)
    x1 = min(bx + bw + 1, lab_image.shape[1])
    y1 = min(by + bh + 1, lab_image.shape[0])
    if x1 <= x or y1 <= y:
        return None

    mask = np.zeros((y1 - y, x1 - x), dtype=np.uint8)
    for seg in ann["segmentation"]:
        pts = np.array(seg, dtype=np.int32).reshape((-1, 2)) - (x, y)
        cv2.fillPoly(mask, [pts], 1)

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None

    radius = max(1, int(round(0.05 * min(mask.shape))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    eroded = cv2.erode(mask, kernel)
    sample = eroded if eroded.any() else mask

    lab_values = lab_image[y:y1, x:x1][sample > 0].astype(np.float32)
    chroma = np.sqrt((lab_values[:, 1] - 128.0) ** 2 + (lab_values[:, 2] - 128.0) ** 2)
    chromatic = lab_values[chroma >= np.percentile(chroma, 75)]

    # return wall, ann_id, cat, route_id, cx, cy (px), area, med_L, med_a, med_b, chroma_p75, chroma_p95
    return {
        "id": ann["id"],
        "image_id": ann["image_id"],
        "category_id": ann["category_id"],
        "route_id": str(ann["attributes"].get("route_id", "")),
        "cx": int(xs.mean()) + x,
        "cy": int(ys.mean()) + y,
        "area": ann["area"],
        "med_L": float(np.median(lab_values[:, 0])),
        "med_a": float(np.median(chromatic[:, 1])),
        "med_b": float(np.median(chromatic[:, 2])),
        "chroma_p75": float(np.percentile(chroma, 75)),
        "chroma_p95": float(np.percentile(chroma, 95)),
    }


def main():
    image_files = sorted([f for f in os.listdir(WHOLE_IMG_PATH) if f.endswith(".jpg") or f.endswith(".png")])
    image_to_annotations = load_annotations(ANNOTATIONS_PATH, image_files)
    image_to_clab_data = defaultdict(list)
    for image_file in image_files:
        image_path = os.path.join(WHOLE_IMG_PATH, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
                
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        for ann in image_to_annotations.get(image_file, []):
            clab_data = extract_clab_from_ann(ann, lab_image)
            if clab_data is not None:
                image_to_clab_data[image_file].append(clab_data)
        
        print(f"Processed {len(image_to_annotations.get(image_file, []))} annotations for image: {image_file}")
    
    # give each route unique route id across all images            
    route_id_map = {}
    next_id = 1
    for image_file, clab_list in image_to_clab_data.items():
        for clab in clab_list:
            route_id = clab["route_id"]
            if route_id not in route_id_map:
                route_id_map[route_id] = next_id
                next_id += 1
            clab["route_id"] = route_id_map[route_id]
        route_id_map.clear()
    
    # save to csv
    all_clab_data = []
    for image_file, clab_list in image_to_clab_data.items():
        for clab in clab_list:
            clab["image_file"] = image_file
            all_clab_data.append(clab)
            
    df = pd.DataFrame(all_clab_data)
    df.to_csv("clab_data.csv", index=False)
    
    print(f"Saved {len(df)} holds, across {next_id - 1} unique routes, to clab_data.csv")
    
    
if __name__ == "__main__":
    main()