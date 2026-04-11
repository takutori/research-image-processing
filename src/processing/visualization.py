from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def draw_bboxes(image: np.ndarray, annotations: list[dict], color: tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    canvas = image.copy()
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    return canvas


def decode_coco_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    return mask.astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    canvas = image.copy().astype(np.float32)
    color_array = np.array(color, dtype=np.float32)
    canvas[mask > 0] = (1 - alpha) * canvas[mask > 0] + alpha * color_array
    return np.clip(canvas, 0, 255).astype(np.uint8)


def draw_keypoints(image: np.ndarray, annotation: dict, color: tuple[int, int, int] = (255, 255, 0), radius: int = 4) -> np.ndarray:
    canvas = image.copy()
    keypoints = annotation.get("keypoints", [])
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i + 3]
        if v > 0:
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color, -1)
    return canvas
