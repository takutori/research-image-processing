from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_result_summary(
    output_dir: Path,
    *,
    image_metrics: dict,
    pixel_metrics: dict,
    extra: dict | None = None,
) -> None:
    payload = {
        "image_level_metrics": _normalize_payload(image_metrics),
        "pixel_level_metrics": _normalize_payload(pixel_metrics),
    }
    if extra:
        payload.update(_normalize_payload(extra))
    save_json(output_dir / "summary.json", payload)


def update_summary_with_test_metrics(
    summary_path: Path,
    *,
    test_image_level_metric: float,
    test_pixel_level_metric: float,
) -> None:
    summary = json.loads(summary_path.read_text())
    summary["test_image_level_metric"] = test_image_level_metric
    summary["test_pixel_level_metric"] = test_pixel_level_metric
    save_json(summary_path, summary)


def _normalize_payload(payload: dict) -> dict:
    normalized = {}
    for key, value in payload.items():
        if isinstance(value, float) and math.isnan(value):
            normalized[key] = None
        else:
            normalized[key] = value
    return normalized
