from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _empty_pose_result() -> dict[str, float]:
    return {
        "ap": 0.0,
        "ap50": 0.0,
        "ap75": 0.0,
        "ap_medium": 0.0,
        "ap_large": 0.0,
        "ar": 0.0,
    }


def compute_coco_pose_ap(coco_gt, dt_results: list[dict], image_ids: list[int] | None = None) -> dict[str, float]:
    if not dt_results:
        return _empty_pose_result()

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(dt_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    if image_ids is not None:
        coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "ap": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "ap_medium": float(stats[3]),
        "ap_large": float(stats[4]),
        "ar": float(stats[5]),
    }


def save_pose_report(
    output_dir: Path,
    ap_results: dict[str, float],
    model_name: str,
    experiment_name: str,
    extra_summary: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_name": experiment_name,
        "model": model_name,
        "ap": ap_results["ap"],
        "ap50": ap_results["ap50"],
        "ap75": ap_results["ap75"],
        "ap_medium": ap_results["ap_medium"],
        "ap_large": ap_results["ap_large"],
        "ar": ap_results["ar"],
    }
    if extra_summary:
        summary.update(extra_summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def save_pose_predictions_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
