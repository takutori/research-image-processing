from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.segmentation.common.data import COCO_ID_TO_NAME


# ---------------------------------------------------------------------------
# Mask conversion
# ---------------------------------------------------------------------------


def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert a binary (H, W) uint8/bool mask to COCO RLE format."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    return mask_util.encode(fortran_mask)


def polygon_to_rle(polygon: list[float], height: int, width: int) -> dict:
    """Convert a flat polygon list to COCO RLE format."""
    rle_list = mask_util.frPyObjects([polygon], height, width)
    return mask_util.merge(rle_list)


# ---------------------------------------------------------------------------
# Mask AP (for YOLO11-seg and Grounded-SAM)
# ---------------------------------------------------------------------------


def compute_coco_mask_ap(
    coco_gt: COCO,
    dt_results: list[dict],
    image_ids: list[int] | None = None,
) -> dict[str, float | list]:
    """Evaluate segmentation masks with pycocotools and return mask AP metrics.

    Args:
        coco_gt: COCO ground truth object.
        dt_results: List of COCO-format segmentation dicts, each with keys
            image_id, category_id, segmentation (RLE), score.
        image_ids: Optional subset of COCO image IDs to evaluate.

    Returns:
        dict with mask_ap, mask_ap50, mask_ap75, and per_category list.
    """
    if not dt_results:
        return _empty_mask_ap_result(coco_gt)

    coco_dt = coco_gt.loadRes(dt_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    if image_ids is not None:
        coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    per_category = _per_category_mask_ap(coco_eval, coco_gt)

    return {
        "mask_ap": float(stats[0]),
        "mask_ap50": float(stats[1]),
        "mask_ap75": float(stats[2]),
        "mask_ap_small": float(stats[3]),
        "mask_ap_medium": float(stats[4]),
        "mask_ap_large": float(stats[5]),
        "per_category": per_category,
    }


def _per_category_mask_ap(coco_eval: COCOeval, coco_gt: COCO) -> list[dict]:
    results = []
    cat_ids = coco_eval.params.catIds
    precision = coco_eval.eval["precision"]  # (T, R, K, A, M)

    for k, cat_id in enumerate(cat_ids):
        ap_vals = precision[:, :, k, 0, 2]
        ap = float(ap_vals[ap_vals > -1].mean()) if (ap_vals > -1).any() else 0.0
        ap50_vals = precision[0, :, k, 0, 2]
        ap50 = float(ap50_vals[ap50_vals > -1].mean()) if (ap50_vals > -1).any() else 0.0
        results.append({
            "category_id": int(cat_id),
            "name": COCO_ID_TO_NAME.get(int(cat_id), str(cat_id)),
            "mask_ap": round(ap, 6),
            "mask_ap50": round(ap50, 6),
        })

    return results


def _empty_mask_ap_result(coco_gt: COCO) -> dict:
    cat_ids = coco_gt.getCatIds()
    per_category = [
        {
            "category_id": int(c),
            "name": COCO_ID_TO_NAME.get(int(c), str(c)),
            "mask_ap": 0.0,
            "mask_ap50": 0.0,
        }
        for c in cat_ids
    ]
    return {
        "mask_ap": 0.0,
        "mask_ap50": 0.0,
        "mask_ap75": 0.0,
        "mask_ap_small": 0.0,
        "mask_ap_medium": 0.0,
        "mask_ap_large": 0.0,
        "per_category": per_category,
    }


# ---------------------------------------------------------------------------
# Best-match IoU + Recall (for SAM 2 automatic mode)
# ---------------------------------------------------------------------------


def compute_recall_and_iou(
    coco_gt: COCO,
    sam2_masks_per_image: dict[int, list[np.ndarray]],
    image_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute best-match IoU and recall for SAM 2 automatic masks.

    For each GT instance mask, find the SAM 2 mask with highest IoU.
    Recall = fraction of GT instances whose best-match IoU >= iou_threshold.

    Args:
        coco_gt: COCO ground truth.
        sam2_masks_per_image: image_id → list of binary (H, W) bool/uint8 masks.
        image_ids: image IDs to evaluate.
        iou_threshold: IoU threshold for recall computation.

    Returns:
        dict with mean_best_iou, recall, n_gt_total, n_matched.
    """
    all_best_ious: list[float] = []
    n_gt_total = 0
    n_matched = 0

    for image_id in image_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=image_id, iscrowd=False)
        anns = coco_gt.loadAnns(ann_ids)
        if not anns:
            continue

        pred_masks = sam2_masks_per_image.get(image_id, [])

        for ann in anns:
            gt_rle = coco_gt.annToRLE(ann)
            gt_mask = mask_util.decode(gt_rle).astype(bool)

            if not pred_masks:
                all_best_ious.append(0.0)
                n_gt_total += 1
                continue

            best_iou = 0.0
            for pred_mask in pred_masks:
                iou = _compute_iou(gt_mask, pred_mask.astype(bool))
                if iou > best_iou:
                    best_iou = iou

            all_best_ious.append(best_iou)
            n_gt_total += 1
            if best_iou >= iou_threshold:
                n_matched += 1

    mean_best_iou = float(np.mean(all_best_ious)) if all_best_ious else 0.0
    recall = n_matched / n_gt_total if n_gt_total > 0 else 0.0

    return {
        "mean_best_iou": round(mean_best_iou, 6),
        f"recall_{int(iou_threshold * 100)}": round(recall, 6),
        "n_gt_total": n_gt_total,
        "n_matched": n_matched,
    }


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float((mask_a & mask_b).sum())
    union = float((mask_a | mask_b).sum())
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


def save_mask_ap_report(
    output_dir: Path,
    ap_results: dict,
    model_name: str,
    experiment_name: str,
) -> None:
    """Save summary.json and per_category_ap.csv for mask AP models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "model": model_name,
        "mask_ap": ap_results["mask_ap"],
        "mask_ap50": ap_results["mask_ap50"],
        "mask_ap75": ap_results["mask_ap75"],
        "mask_ap_small": ap_results["mask_ap_small"],
        "mask_ap_medium": ap_results["mask_ap_medium"],
        "mask_ap_large": ap_results["mask_ap_large"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    per_cat_df = pd.DataFrame(ap_results["per_category"])
    per_cat_df.to_csv(output_dir / "per_category_ap.csv", index=False)


def save_sam2_report(
    output_dir: Path,
    recall_results: dict,
    per_image_rows: list[dict],
    model_name: str,
    experiment_name: str,
) -> None:
    """Save summary.json and predictions.csv for SAM 2."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "model": model_name,
        "mean_best_iou": recall_results["mean_best_iou"],
        "recall_50": recall_results.get("recall_50", 0.0),
        "n_gt_total": recall_results["n_gt_total"],
        "n_matched": recall_results["n_matched"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    pd.DataFrame(per_image_rows).to_csv(output_dir / "predictions.csv", index=False)


class _JSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types and RLE bytes."""

    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_predictions_json(output_dir: Path, dt_results: list[dict]) -> None:
    """Save raw COCO-format segmentation predictions as predictions.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(
        json.dumps(dt_results, cls=_JSONEncoder, ensure_ascii=False, indent=2)
    )
