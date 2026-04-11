from __future__ import annotations

import cv2
import numpy as np


def convert_bgr_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def convert_bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def apply_binary_threshold(image_gray: np.ndarray, threshold: int = 140) -> tuple[np.ndarray, np.ndarray]:
    _, binary = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
    _, binary_inv = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary, binary_inv


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_median_blur(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    return cv2.medianBlur(image, kernel_size)


def normalize_image(image: np.ndarray, mean: tuple[float, float, float] = (0.485, 0.456, 0.406), std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> tuple[np.ndarray, np.ndarray]:
    image_float = image.astype(np.float32) / 255.0
    mean_array = np.array(mean, dtype=np.float32)
    std_array = np.array(std, dtype=np.float32)
    normalized = (image_float - mean_array) / std_array
    normalized_for_display = np.clip(
        (normalized - normalized.min()) / (normalized.max() - normalized.min()),
        0,
        1,
    )
    return normalized, normalized_for_display
