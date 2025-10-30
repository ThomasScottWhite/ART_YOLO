import os
import json
from PIL import Image
from pathlib import Path

def save_split(split_name, samples, output_dir: Path):
    image_dir = output_dir / split_name
    os.makedirs(image_dir, exist_ok=True)

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Save images and collect category IDs
    category_set = set()
    for sample in samples:
        image_id = sample["image_id"]
        image = sample["image"]
        image.save(image_dir / f"{image_id}.jpg")
        category_set.update(sample["objects"]["category"])

    # Create category map
    category_map = {cat_id: idx for idx, cat_id in enumerate(sorted(category_set))}

    for cat_id, new_id in category_map.items():
        coco_dict["categories"].append({
            "id": new_id,
            "name": str(cat_id),
            "supercategory": "none"
        })

    ann_id = 1
    for sample in samples:
        image_id = sample["image_id"]
        width = sample["width"]
        height = sample["height"]

        coco_dict["images"].append({
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "width": width,
            "height": height
        })

        for bbox, cat in zip(sample["objects"]["bbox"], sample["objects"]["category"]):
            x_min, y_min, x_max, y_max = bbox
            box_w = x_max - x_min
            box_h = y_max - y_min

            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_map[cat],
                "bbox": [x_min, y_min, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0
            })
            ann_id += 1

    with open(output_dir / f"{split_name}_annotations.json", "w") as f:
        json.dump(coco_dict, f, indent=2)

def save_dataset_to_file(train, val):
    output_dir = Path("./dataset/dfine_coco/")
    os.makedirs(output_dir, exist_ok=True)

    save_split("train", train, output_dir)
    save_split("val", val, output_dir)
