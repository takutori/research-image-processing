from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.object_detection.common.data import COCO_ID_TO_NAME


def compute_coco_map(
    coco_gt: COCO,
    dt_results: list[dict],
    image_ids: list[int] | None = None,
) -> dict[str, float | list]:
    """Evaluate detections with pycocotools and return mAP metrics.

    Args:
        coco_gt: COCO ground truth object.
        dt_results: List of COCO-format detection dicts, each with keys
            image_id, category_id, bbox ([x, y, w, h]), score.
        image_ids: Optional subset of COCO image IDs to evaluate.

    Returns:
        dict with map, map50, map75, map_small, map_medium, map_large,
        and per_category list of {category_id, name, ap, ap50}.
    """
    if not dt_results:
        return _empty_map_result(coco_gt)

    coco_dt = coco_gt.loadRes(dt_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if image_ids is not None:
        coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    per_category = _per_category_ap(coco_eval, coco_gt)

    return {
        "map": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
        "per_category": per_category,
    }


def _per_category_ap(coco_eval: COCOeval, coco_gt: COCO) -> list[dict]:
    """Extract per-category AP@[0.5:0.95] and AP@0.5."""
    results = []
    cat_ids = coco_eval.params.catIds
    # precision shape: (T=10, R=101, K=num_cats, A=4, M=3)
    precision = coco_eval.eval["precision"]

    for k, cat_id in enumerate(cat_ids):
        # AP@[0.5:0.95]: average over all IoU thresholds, area=all, maxDets=100
        ap_vals = precision[:, :, k, 0, 2]
        ap = float(ap_vals[ap_vals > -1].mean()) if (ap_vals > -1).any() else 0.0
        # AP@0.5: IoU threshold index 0 corresponds to 0.5
        ap50_vals = precision[0, :, k, 0, 2]
        ap50 = float(ap50_vals[ap50_vals > -1].mean()) if (ap50_vals > -1).any() else 0.0

        results.append({
            "category_id": int(cat_id),
            "name": COCO_ID_TO_NAME.get(int(cat_id), str(cat_id)),
            "ap": round(ap, 6),
            "ap50": round(ap50, 6),
        })

    return results


def _empty_map_result(coco_gt: COCO) -> dict:
    cat_ids = coco_gt.getCatIds()
    per_category = [
        {
            "category_id": int(c),
            "name": COCO_ID_TO_NAME.get(int(c), str(c)),
            "ap": 0.0,
            "ap50": 0.0,
        }
        for c in cat_ids
    ]
    return {
        "map": 0.0,
        "map50": 0.0,
        "map75": 0.0,
        "map_small": 0.0,
        "map_medium": 0.0,
        "map_large": 0.0,
        "per_category": per_category,
    }


def save_detection_report(
    output_dir: Path,
    map_results: dict,
    model_name: str,
    experiment_name: str,
) -> None:
    """Save summary.json and per_category_ap.csv to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "model": model_name,
        "map": map_results["map"],
        "map50": map_results["map50"],
        "map75": map_results["map75"],
        "map_small": map_results["map_small"],
        "map_medium": map_results["map_medium"],
        "map_large": map_results["map_large"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    per_cat_df = pd.DataFrame(map_results["per_category"])
    per_cat_df.to_csv(output_dir / "per_category_ap.csv", index=False)


def save_predictions_json(output_dir: Path, dt_results: list[dict]) -> None:
    """Save raw COCO-format predictions as predictions.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(
        json.dumps(dt_results, ensure_ascii=False, indent=2)
    )
