import os
import cv2
from collections import defaultdict
import json

import numpy as np
import pandas as pd

WHOLE_IMG_PATH = "/home/public/rwiddop/images/"
ANNOTATIONS_PATH = "/home/public/rwiddop/annotations.json"
VOLUME_CATEGORY_ID = 1
HUE_BINS = 16
CHROMA_FLOOR = 8.0
HUE_COLS = [f"hue_{k:02d}" for k in range(HUE_BINS)]


def load_annotations(annotations_path, image_files):
    with open(annotations_path, "r") as f:
        data = json.load(f)
    name_to_id = {img["file_name"]: img["id"] for img in data["images"]}
    wanted = {name_to_id[f]: f for f in image_files if f in name_to_id}
    image_to_annotations = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] in wanted:
            image_to_annotations[wanted[ann["image_id"]]].append(ann)
    for cat in data["categories"]:
        if cat["name"].lower() == "volume":
            volume_category_id = cat["id"]
            break
    return image_to_annotations, volume_category_id


def hue_hist(lab_values, nbins=HUE_BINS, floor=CHROMA_FLOOR):
    a = lab_values[:, 1] - 128.0
    b = lab_values[:, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    keep = chroma >= floor
    if keep.sum() < 16:
        return np.full(nbins, 1.0 / nbins), 0.0

    w = chroma[keep]
    pos = (np.arctan2(b[keep], a[keep]) + np.pi) / (2 * np.pi) * nbins
    lo = np.floor(pos).astype(int)
    frac = pos - lo
    hist = np.zeros(nbins)
    np.add.at(hist, lo % nbins, w * (1.0 - frac))
    np.add.at(hist, (lo + 1) % nbins, w * frac)
    return hist / hist.sum(), float(keep.mean())


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
    hist, hue_conf = hue_hist(lab_values)

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
        "hue_conf": hue_conf,
        **{f"hue_{k:02d}": float(v) for k, v in enumerate(hist)},
    }


def circular_emd(a, b):
    c = np.cumsum(a - b)
    return float(np.abs(c - np.median(c)).sum())


def create_pair(i, j):
    hi = np.array([i[col] for col in HUE_COLS], dtype=np.float32)
    hj = np.array([j[col] for col in HUE_COLS], dtype=np.float32)
    pair = {
        "i_id": i["id"],
        "j_id": j["id"],
        "image_file": i["image_file"],
        "dx": abs(i["cx"] - j["cx"]),
        "dy": abs(i["cy"] - j["cy"]),
        "distance": np.sqrt((i["cx"] - j["cx"])**2 + (i["cy"] - j["cy"])**2),
        "L_diff": abs(i["med_L"] - j["med_L"]),
        "a_diff": abs(i["med_a"] - j["med_a"]),
        "b_diff": abs(i["med_b"] - j["med_b"]),
        "distance_ab": np.sqrt((i["med_a"] - j["med_a"])**2 + (i["med_b"] - j["med_b"])**2),
        "c75_diff": abs(i["chroma_p75"] - j["chroma_p75"]),
        "c95_diff": abs(i["chroma_p95"] - j["chroma_p95"]),
        "chroma_max": max(i["chroma_p75"], j["chroma_p75"]),
        "chroma_min": min(i["chroma_p75"], j["chroma_p75"]),
        "hue_emd": circular_emd(hi, hj),
        "hue_conf_diff": abs(i["hue_conf"] - j["hue_conf"]),
        "hue_conf_max": max(i["hue_conf"], j["hue_conf"]),
        "hue_conf_min": min(i["hue_conf"], j["hue_conf"]),
        "cat_min": min(i["category_id"], j["category_id"]),
        "cat_max": max(i["category_id"], j["category_id"]),
        "same_cat": int(i["category_id"] == j["category_id"]),
        "area_ratio": min(i["area"], j["area"]) / max(i["area"], j["area"]),
        "absolute_area": abs(i["area"] - j["area"]),
        "same_route": int(i["route_id"] == j["route_id"]),
        "i_route": i["route_id"],
        "j_route": j["route_id"],
    }
    return pair


def main():
    image_files = sorted([f for f in os.listdir(WHOLE_IMG_PATH) if f.endswith(".jpg") or f.endswith(".png")])
    image_to_annotations, volume_category_id = load_annotations(ANNOTATIONS_PATH, image_files)
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
        
        print(f"{image_file}\t| {len(image_to_clab_data.get(image_file, []))} holds processed")
    
    # give each route unique route id across all images, ignore volumes          
    route_id_map = {}
    next_id = 1
    for image_file, clab_list in image_to_clab_data.items():
        for clab in clab_list:
            if clab["category_id"] == volume_category_id:
                continue
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
            
    dataset = pd.DataFrame(all_clab_data)
    dataset.to_csv("clab_data.csv", index=False)
    
    print(f"Saved {len(dataset)} holds, across {next_id - 1} unique routes, to clab_data.csv")
    print(f"Volume category id: {volume_category_id}")

    datasets_grouped = dataset.groupby("image_file")
    image_to_data = {image_file: data for image_file, data in datasets_grouped}

    image_to_pairs = defaultdict(list)
    for image_file, data in image_to_data.items():
        data = data[
            (data["category_id"] != VOLUME_CATEGORY_ID)
            & data["route_id"].notna()
            & (data["route_id"].astype(str).str.strip() != "")
        ].copy()
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                pair = create_pair(data.iloc[i], data.iloc[j])
                image_to_pairs[image_file].append(pair)
        print(f"Image File: {image_file}, Pairs: {len(image_to_pairs[image_file])}")
        
    all_pairs = []
    for image_file, pairs in image_to_pairs.items():
        all_pairs.extend(pairs)
    all_pairs_df = pd.DataFrame(all_pairs)
    # save all pairs DataFrame to a CSV file
    all_pairs_df.to_csv("all_pairs.csv", index=False)
    
    print(f"All Pairs DataFrame Shape: {all_pairs_df.shape}")
    print(f"Total Pairs: {len(all_pairs)}")
    

if __name__ == "__main__":
    main()
