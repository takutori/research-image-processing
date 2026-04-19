from __future__ import annotations

import json
import os
from pathlib import Path

COCO_PERSON_CATEGORY = {"id": 1, "name": "person"}
COCO_PERSON_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
COCO_PERSON_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def resolve_pose_dataset_root(dataset_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    base = Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"
    return base / dataset_name


def load_coco_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def build_coco_gt(ann_file: Path | str):
    from pycocotools.coco import COCO

    return COCO(str(ann_file))


def get_image_ids(coco_gt, max_images: int | None = None) -> list[int]:
    image_ids = sorted(coco_gt.getImgIds())
    if max_images is not None:
        image_ids = image_ids[:max_images]
    return image_ids


def get_image_path(coco_gt, image_id: int, images_dir: Path) -> Path:
    info = coco_gt.loadImgs(image_id)[0]
    return images_dir / info["file_name"]


def group_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for annotation in annotations:
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    return grouped


def get_detection_lookup(detections: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for detection in detections:
        grouped.setdefault(int(detection["image_id"]), []).append(detection)
    return grouped


def flatten_keypoints_with_scores(keypoints_xy: list[list[float]], keypoint_scores: list[float]) -> list[float]:
    flattened: list[float] = []
    for xy, score in zip(keypoints_xy, keypoint_scores):
        flattened.extend([float(xy[0]), float(xy[1]), float(score)])
    return flattened
