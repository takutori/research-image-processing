from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.common import (
    build_visa_datamodule,
    build_category_metrics,
    build_threshold_curve,
    compute_image_level_metrics,
    compute_pixel_level_metrics,
    get_prediction_field,
    iter_prediction_items,
    list_visa_categories,
    save_dataframe,
    save_json,
    save_result_summary,
    setup_visa_pytorch,
    to_numpy,
    update_summary_with_test_metrics,
)
from src.anomaly_detection.common.data import setup_visa_pytorch
from src.anomaly_detection.winclip_fewshot.config import DEFAULT_CONFIG_PATH, load_experiment_config
from src.anomaly_detection.winclip_fewshot.model import build_winclip_fewshot_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WinCLIP few-shot on VisA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def get_categories(config) -> list[str]:
    return config.categories or list_visa_categories()


def build_datamodule(config, *, category: str, reference_dir: Path):
    from anomalib.data import Folder
    from torchvision.transforms.v2 import Resize

    dataset_root = Path(config.dataset_root)
    visa_root = dataset_root / "visa_pytorch" / category
    resize = Resize((240, 240))
    return Folder(
        name=f"visa_smoke_{category}",
        root=visa_root,
        normal_dir=reference_dir,
        normal_test_dir=visa_root / "test" / "good",
        abnormal_dir=visa_root / "test" / "bad",
        mask_dir=visa_root / "ground_truth" / "bad",
        train_batch_size=config.test.batch_size,
        eval_batch_size=config.test.batch_size,
        num_workers=config.test.num_workers,
        augmentations=resize,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
    )


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    setup_visa_pytorch(config.dataset_root)

    summary_path = config.output_path / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"experiment_summary.json not found: {summary_path}. Run train.py first.")
    summary = json.loads(summary_path.read_text())
    model_params = summary.get("best_model_config", {})
    reference_root = Path(model_params.get("few_shot_source", config.output_path / "reference_images"))

    output_dir = config.output_path / "best_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    category_frames: list[pd.DataFrame] = []
    pixel_rows: list[dict] = []

    for category in get_categories(config):
        few_shot_source = reference_root / category
        datamodule = build_datamodule(config, category=category, reference_dir=few_shot_source)
        model = build_winclip_fewshot_model(
            class_name=category,
            scales=tuple(model_params.get("scales", config.model.scales)),
            k_shot=int(model_params.get("k_shot", config.model.k_shot)),
        )
        run_dir = output_dir / category / "anomalib_run"
        from anomalib.engine import Engine
        engine = Engine(default_root_dir=str(run_dir), accelerator=accelerator, devices=1)
        predictions = engine.predict(model=model, datamodule=datamodule, return_predictions=True)
        pred_df, anomaly_maps, masks = _parse_predictions(predictions, category=category)
        category_frames.append(pred_df)
        pixel_rows.append({"category": category, **compute_pixel_level_metrics(anomaly_maps, masks)})

    predictions_df = pd.concat(category_frames, ignore_index=True)
    image_metrics = compute_image_level_metrics(predictions_df)
    category_metrics_df = build_category_metrics(predictions_df)
    pixel_df = pd.DataFrame(pixel_rows)
    valid_pixel = pixel_df.dropna(subset=["pixel_auroc"])
    pixel_auroc = float(valid_pixel["pixel_auroc"].mean()) if len(valid_pixel) else float("nan")
    pixel_ap = float(valid_pixel["pixel_ap"].mean()) if len(valid_pixel) else float("nan")
    threshold_curve_df = build_threshold_curve(predictions_df)

    save_dataframe(output_dir / "predictions.csv", predictions_df)
    save_dataframe(output_dir / "category_metrics.csv", category_metrics_df)
    save_dataframe(output_dir / "threshold_curve.csv", threshold_curve_df)
    save_json(output_dir / "image_level_metrics.json", image_metrics)
    pixel_metrics = {"pixel_auroc": pixel_auroc, "pixel_ap": pixel_ap}
    save_json(output_dir / "pixel_level_metrics.json", pixel_metrics)
    save_result_summary(output_dir, image_metrics=image_metrics, pixel_metrics=pixel_metrics, extra=model_params)

    update_summary_with_test_metrics(
        summary_path,
        test_image_level_metric=float(image_metrics["image_auroc"]),
        test_pixel_level_metric=pixel_auroc,
    )
    print(f"evaluation completed. results saved to: {output_dir}")


def _parse_predictions(predictions, *, category: str) -> tuple[pd.DataFrame, list, list]:
    import numpy as np

    rows = []
    anomaly_maps = []
    masks = []

    for item in iter_prediction_items(predictions):
        image_path = get_prediction_field(item, "image_path")
        pred_score = to_numpy(get_prediction_field(item, "pred_score"))
        anomaly_map = to_numpy(get_prediction_field(item, "anomaly_map"))
        gt_mask_value = get_prediction_field(item, "gt_mask")
        if gt_mask_value is None:
            gt_mask_value = get_prediction_field(item, "mask")
        gt_label_value = get_prediction_field(item, "gt_label")
        if gt_label_value is None:
            gt_label_value = get_prediction_field(item, "label")
        gt_mask = to_numpy(gt_mask_value)
        gt_label = to_numpy(gt_label_value)

        if pred_score is None and anomaly_map is not None:
            score = float(np.asarray(anomaly_map).max())
        elif pred_score is not None:
            score = float(np.asarray(pred_score).reshape(-1)[0])
        else:
            score = 0.0

        if gt_label is not None:
            is_anomaly = bool(np.asarray(gt_label).astype(float).max() > 0)
        else:
            is_anomaly = bool(gt_mask is not None and np.asarray(gt_mask).sum() > 0)

        rows.append({"image_path": str(image_path) if image_path else None, "category": category, "is_anomaly": is_anomaly, "anomaly_score": score})
        if anomaly_map is not None and gt_mask is not None:
            anomaly_maps.append(np.asarray(anomaly_map).squeeze())
            masks.append(np.asarray(gt_mask).squeeze())

    return pd.DataFrame(rows), anomaly_maps, masks


if __name__ == "__main__":
    main()
