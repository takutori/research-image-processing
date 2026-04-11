from __future__ import annotations

import json
from pathlib import Path


def load_coco_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_image_index(coco: dict) -> dict[int, dict]:
    return {image["id"]: image for image in coco["images"]}


def build_category_index(coco: dict) -> dict[int, dict]:
    return {category["id"]: category for category in coco["categories"]}


def group_annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        grouped.setdefault(annotation["image_id"], []).append(annotation)
    return grouped


def select_image_with_min_annotations(coco: dict, annotations_by_image: dict[int, list[dict]], min_annotations: int = 3) -> tuple[dict, list[dict]]:
    for image in coco["images"]:
        annotations = annotations_by_image.get(image["id"], [])
        if len(annotations) >= min_annotations:
            return image, annotations
    raise ValueError(f"no image found with at least {min_annotations} annotations")
