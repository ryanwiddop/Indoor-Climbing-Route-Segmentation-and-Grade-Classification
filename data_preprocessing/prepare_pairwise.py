import numpy as np
import pandas as pd
from collections import defaultdict

VOLUME_CATEGORY_ID = 1

def calculate_wall_scale(holds):
    hold_sizes = [
        np.sqrt(hold["area"])
        for hold in holds
        if hold["category_id"] != VOLUME_CATEGORY_ID
    ]

    if not hold_sizes:
        raise ValueError("Cannot calculate wall scale without regular holds")

    return float(np.median(hold_sizes))


def create_pair(i, j, wall_scale):
    pair = {
        "i_id": i["id"],
        "j_id": j["id"],
        "image_file": i["image_file"],
        "dx_normalized": abs(i["cx"] - j["cx"]) / wall_scale,
        "dy_normalized": abs(i["cy"] - j["cy"]) / wall_scale,
        "distance_normalized": np.sqrt((i["cx"] - j["cx"])**2 + (i["cy"] - j["cy"])**2) / wall_scale,
        "L_diff": abs(i["med_L"] - j["med_L"]),
        "a_diff": abs(i["med_a"] - j["med_a"]),
        "b_diff": abs(i["med_b"] - j["med_b"]),
        "distance_ab": np.sqrt((i["med_a"] - j["med_a"])**2 + (i["med_b"] - j["med_b"])**2),
        "c75_diff": abs(i["chroma_p75"] - j["chroma_p75"]),
        "c95_diff": abs(i["chroma_p95"] - j["chroma_p95"]),
        "chroma_max": max(i["chroma_p75"], j["chroma_p75"]),
        "chroma_min": min(i["chroma_p75"], j["chroma_p75"]),
        "cat_min": min(i["category_id"], j["category_id"]),
        "cat_max": max(i["category_id"], j["category_id"]),
        "same_cat": int(i["category_id"] == j["category_id"]),
        "area_ratio": min(i["area"], j["area"]) / max(i["area"], j["area"]),
        "absolute_area": abs(i["area"] - j["area"]) / wall_scale**2,
        "same_route": int(i["route_id"] == j["route_id"]),
        "i_route": i["route_id"],
        "j_route": j["route_id"]
    }
    return pair


def main():
    dataset = pd.read_csv("clab_data.csv")
    datasets_grouped = dataset.groupby("image_file")
    image_to_data = {image_file: data for image_file, data in datasets_grouped}

    image_to_pairs = defaultdict(list)
    for image_file, data in image_to_data.items():
        data = data[
            (data["category_id"] != VOLUME_CATEGORY_ID)
            & data["route_id"].notna()
            & (data["route_id"].astype(str).str.strip() != "")
        ].copy()
        wall_scale = calculate_wall_scale([data.iloc[i] for i in range(len(data))])
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                pair = create_pair(data.iloc[i], data.iloc[j], wall_scale)
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