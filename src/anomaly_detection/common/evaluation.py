from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_image_level_metrics(
    df: pd.DataFrame,
    *,
    label_column: str = "is_anomaly",
    score_column: str = "anomaly_score",
) -> dict[str, float]:
    labels = df[label_column].astype(int).to_numpy()
    scores = df[score_column].to_numpy()
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"image_auroc": float("nan"), "image_ap": float("nan"), "best_f1": float("nan"), "best_threshold": float("nan"), "num_examples": int(len(df)), "positive_ratio": float(labels.mean())}
    auroc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    best_f1, best_thresh = _find_best_f1(labels, scores)
    return {
        "image_auroc": auroc,
        "image_ap": ap,
        "best_f1": best_f1,
        "best_threshold": best_thresh,
        "num_examples": int(len(df)),
        "positive_ratio": float(labels.mean()),
    }


def _find_best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_f1, best_thresh = 0.0, float(thresholds[0])
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_f1, best_thresh


def compute_pixel_level_metrics(
    anomaly_maps: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict[str, float]:
    if not anomaly_maps or not masks:
        return {"pixel_auroc": float("nan"), "pixel_ap": float("nan")}
    flat_scores = np.concatenate([m.flatten() for m in anomaly_maps])
    flat_labels = np.concatenate([m.flatten().astype(int) for m in masks])
    if flat_labels.sum() == 0:
        return {"pixel_auroc": float("nan"), "pixel_ap": float("nan")}
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_scores)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_scores)),
    }


def build_category_metrics(
    predictions_df: pd.DataFrame,
    *,
    train_size_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for category, cat_df in predictions_df.groupby("category", sort=True):
        metrics = compute_image_level_metrics(cat_df)
        rows.append({"category": category, "num_test_examples": int(len(cat_df)), **metrics})
    df = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
    if train_size_df is not None and len(train_size_df) > 0:
        df = df.merge(train_size_df[["category", "num_train_normal_images"]], on="category", how="left")
    return df


def build_threshold_curve(predictions_df: pd.DataFrame) -> pd.DataFrame:
    labels = predictions_df["is_anomaly"].astype(int).to_numpy()
    scores = predictions_df["anomaly_score"].to_numpy()
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    rows = []
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"threshold": float(thresh), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)
