from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.anomaly_detection.common.evaluation import compute_image_level_metrics, compute_pixel_level_metrics


@dataclass
class PredictionOutput:
    predictions: pd.DataFrame
    anomaly_maps: list[np.ndarray]
    masks: list[np.ndarray]


def build_prediction_output(
    *,
    image_paths: list[str | None],
    categories: list[str],
    labels: np.ndarray,
    masks: list[np.ndarray | None],
    pred_scores: np.ndarray,
    anomaly_maps: np.ndarray,
) -> PredictionOutput:
    rows = []
    pixel_maps: list[np.ndarray] = []
    pixel_masks: list[np.ndarray] = []
    pred_scores = np.asarray(pred_scores).reshape(-1)
    anomaly_maps = np.asarray(anomaly_maps)

    for index, score in enumerate(pred_scores):
        label = int(labels[index]) if index < len(labels) else 0
        rows.append(
            {
                "image_path": image_paths[index] if index < len(image_paths) else None,
                "category": categories[index] if index < len(categories) else None,
                "is_anomaly": bool(label > 0),
                "anomaly_score": float(score),
            }
        )

        mask = masks[index] if index < len(masks) else None
        if mask is not None and index < len(anomaly_maps):
            pixel_maps.append(_squeeze_map(anomaly_maps[index], target_shape=np.asarray(mask).squeeze().shape))
            pixel_masks.append(np.asarray(mask).squeeze())

    return PredictionOutput(pd.DataFrame(rows), pixel_maps, pixel_masks)


def summarize_predictions(output: PredictionOutput) -> dict[str, float]:
    image_metrics = compute_image_level_metrics(output.predictions)
    pixel_metrics = compute_pixel_level_metrics(output.anomaly_maps, output.masks)
    pixel_f1 = _compute_pixel_best_f1(output.anomaly_maps, output.masks)
    return {**image_metrics, **pixel_metrics, **pixel_f1}


def _squeeze_map(anomaly_map: np.ndarray, *, target_shape: tuple[int, ...]) -> np.ndarray:
    anomaly_map = np.asarray(anomaly_map).squeeze()
    if len(target_shape) == 2 and anomaly_map.shape != target_shape:
        import cv2

        anomaly_map = cv2.resize(anomaly_map.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return anomaly_map


def _compute_pixel_best_f1(anomaly_maps: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    if not anomaly_maps or not masks:
        return {"pixel_best_f1": float("nan"), "pixel_best_threshold": float("nan")}
    scores = np.concatenate([np.asarray(x).reshape(-1) for x in anomaly_maps])
    labels = np.concatenate([np.asarray(x).astype(int).reshape(-1) for x in masks])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"pixel_best_f1": float("nan"), "pixel_best_threshold": float("nan")}
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 200)
    best_f1 = 0.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return {"pixel_best_f1": best_f1, "pixel_best_threshold": best_threshold}
