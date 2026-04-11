from __future__ import annotations

import cv2
import numpy as np


def resize_image_and_bboxes(image: np.ndarray, annotations: list[dict], target_width: int, target_height: int) -> tuple[np.ndarray, list[dict]]:
    original_height, original_width = image.shape[:2]
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    resized_annotations = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        resized_annotations.append({**ann, "bbox": [x * scale_x, y * scale_y, w * scale_x, h * scale_y]})
    return resized_image, resized_annotations


def center_crop_image_and_bboxes(image: np.ndarray, annotations: list[dict], crop_width: int, crop_height: int) -> tuple[np.ndarray, list[dict]]:
    original_height, original_width = image.shape[:2]
    crop_width = min(crop_width, original_width)
    crop_height = min(crop_height, original_height)
    crop_x0 = max((original_width - crop_width) // 2, 0)
    crop_y0 = max((original_height - crop_height) // 2, 0)

    cropped_image = image[crop_y0:crop_y0 + crop_height, crop_x0:crop_x0 + crop_width]
    cropped_annotations = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1 = np.clip(x - crop_x0, 0, crop_width)
        y1 = np.clip(y - crop_y0, 0, crop_height)
        x2 = np.clip(x + w - crop_x0, 0, crop_width)
        y2 = np.clip(y + h - crop_y0, 0, crop_height)
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w >= 1 and new_h >= 1:
            cropped_annotations.append({**ann, "bbox": [float(x1), float(y1), float(new_w), float(new_h)]})
    return cropped_image, cropped_annotations


def pad_to_square_image_and_bboxes(image: np.ndarray, annotations: list[dict], fill_value: tuple[int, int, int] = (0, 0, 0)) -> tuple[np.ndarray, list[dict]]:
    height, width = image.shape[:2]
    pad_target = max(height, width)
    pad_top = (pad_target - height) // 2
    pad_bottom = pad_target - height - pad_top
    pad_left = (pad_target - width) // 2
    pad_right = pad_target - width - pad_left

    padded_image = cv2.copyMakeBorder(
        image,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=fill_value,
    )
    padded_annotations = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        padded_annotations.append({**ann, "bbox": [x + pad_left, y + pad_top, w, h]})
    return padded_image, padded_annotations


def horizontal_flip_image_and_bboxes(image: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, list[dict]]:
    width = image.shape[1]
    flipped_image = cv2.flip(image, 1)
    flipped_annotations = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        flipped_annotations.append({**ann, "bbox": [width - x - w, y, w, h]})
    return flipped_image, flipped_annotations
