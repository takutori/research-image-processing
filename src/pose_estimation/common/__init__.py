"""Common utilities for pose estimation tasks."""

from .data import (
    COCO_PERSON_CATEGORY,
    COCO_PERSON_FLIP_IDX,
    COCO_PERSON_KEYPOINT_NAMES,
    build_coco_gt,
    flatten_keypoints_with_scores,
    get_detection_lookup,
    get_image_ids,
    get_image_path,
    group_annotations_by_image,
    load_coco_json,
    resolve_pose_dataset_root,
)
from .evaluation import (
    compute_coco_pose_ap,
    save_pose_predictions_csv,
    save_pose_report,
)

__all__ = [
    "COCO_PERSON_CATEGORY",
    "COCO_PERSON_FLIP_IDX",
    "COCO_PERSON_KEYPOINT_NAMES",
    "build_coco_gt",
    "compute_coco_pose_ap",
    "flatten_keypoints_with_scores",
    "get_detection_lookup",
    "get_image_ids",
    "get_image_path",
    "group_annotations_by_image",
    "load_coco_json",
    "resolve_pose_dataset_root",
    "save_pose_predictions_csv",
    "save_pose_report",
]
