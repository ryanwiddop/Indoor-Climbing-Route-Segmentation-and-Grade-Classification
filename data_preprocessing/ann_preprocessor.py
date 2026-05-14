"""
Validate and normalize a CVAT COCO export.

Input:  CVAT export (instances_default.json) - full-wall polygon annotations.
Output: walls_clean.json - same schema with:
  - 'occluded' attribute stripped
  - route_id coerced to int where possible
  - route_grade validated against VALID_GRADES (invalid values nulled out)
  - per-route conflict reporting (same route_id with multiple grades on one image)
  - grade distribution chart written to figures/grade_distribution.png

Usage:
    python ann_preprocessor.py <input.json> <output.json>
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID_GRADES = {
    "VB", "V0", "V1", "V2", "V3", "V4", "V5", "V6",
    "V7", "V8", "V9", "V10", "V11", "V12", "INC",
}
GRADE_ORDER = [
    "VB", "V0", "V1", "V2", "V3", "V4", "V5", "V6",
    "V7", "V8", "V9", "V10", "V11", "V12", "INC",
]


def parse_route_id(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def parse_grade(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"undefined", "null", "none", "nan"}:
        return None
    return s


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    with open(args.input) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_name = {im["id"]: im["file_name"] for im in coco["images"]}

    hold_counts = 0
    invalid_grades = []
    cleaned_annotations = []
    route_grades_by_image = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for ann in coco["annotations"]:
        hold_counts += 1
        img_id = ann["image_id"]
        img_name = img_id_to_name.get(img_id, f"<id={img_id}>")
        attr = ann.get("attributes", {})

        route_id = parse_route_id(attr.get("route_id"))
        grade = parse_grade(attr.get("route_grade"))
        cat_name = cat_id_to_name.get(ann["category_id"], "unknown")
        is_volume = cat_name == "volume"

        if grade is not None and grade not in VALID_GRADES:
            invalid_grades.append((img_name, ann["id"], grade))
            grade = None

        if not is_volume:
            if route_id is None:
                print(f"{img_name}:{ann['id']} - Missing route_id (non-volume)")
            if grade is None:
                print(f"{img_name}:{ann['id']}\tID:{route_id} - Missing route_grade")

        if route_id is not None:
            route_grades_by_image[img_id][route_id][grade].append(ann["id"])

        new_attr = {}
        if route_id is not None:
            new_attr["route_id"] = route_id
        if grade is not None:
            new_attr["route_grade"] = grade

        new_ann = dict(ann)
        new_ann["attributes"] = new_attr
        cleaned_annotations.append(new_ann)

    for img_id, route_map in route_grades_by_image.items():
        img_name = img_id_to_name[img_id]
        for route_id, grade_to_ann_ids in route_map.items():
            grades_clean = {g for g in grade_to_ann_ids.keys() if g is not None}
            if len(grades_clean) > 1:
                print(f"{img_name} - Route {route_id} has conflicting grades:")
                for g in sorted(grades_clean):
                    print(f"    {g}: ann_ids={sorted(grade_to_ann_ids[g])}")
                majority = max(len(ids) for g, ids in grade_to_ann_ids.items() if g is not None)
                outliers = [
                    (g, ids) for g, ids in grade_to_ann_ids.items()
                    if g is not None and len(ids) < majority
                ]
                if outliers:
                    print("  Outliers:")
                    for g, ids in sorted(outliers, key=lambda x: str(x[0])):
                        print(f"    {sorted(ids)} -> {g}")

    grade_distribution = defaultdict(int)
    seen_routes = set()
    for img_id, route_map in route_grades_by_image.items():
        for route_id, grade_to_ann_ids in route_map.items():
            grades_clean = {g for g in grade_to_ann_ids.keys() if g is not None}
            if not grades_clean:
                continue
            chosen = max(grades_clean, key=lambda g: (len(grade_to_ann_ids[g]), g))
            grade_distribution[chosen] += 1
            seen_routes.add((img_id, route_id))

    if invalid_grades:
        print(f"\n{len(invalid_grades)} annotations had invalid grades - set to None:")
        for img_name, ann_id, grade in invalid_grades[:20]:
            print(f"  {img_name}:{ann_id} - {grade!r}")

    cleaned = dict(coco)
    cleaned["annotations"] = cleaned_annotations
    with open(args.output, "w") as f:
        json.dump(cleaned, f)
    print(f"\nCleaned COCO written to {args.output}")
    print(f"Total holds       : {hold_counts}")
    print(f"Total routes      : {len(seen_routes)}")
    print(f"Total images      : {len(coco['images'])}")
    print("Grade distribution:")
    for g in GRADE_ORDER:
        if g in grade_distribution:
            print(f"  {g}: {grade_distribution[g]}")

    ordered = [g for g in GRADE_ORDER if g in grade_distribution]
    counts = [grade_distribution[g] for g in ordered]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ordered, counts)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Route Count")
    ax.set_title("Grade Distribution")
    plt.tight_layout()
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "grade_distribution.png")
    fig.savefig(fig_path)
    print(f"Grade distribution chart saved to {fig_path}")


if __name__ == "__main__":
    main()
