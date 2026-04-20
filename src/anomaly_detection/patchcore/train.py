from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.common import (
    build_visa_synthetic_datamodule,
    build_category_metrics,
    build_threshold_curve,
    compute_pixel_level_metrics,
    get_prediction_field,
    iter_prediction_items,
    list_visa_categories,
    save_dataframe,
    save_json,
    save_result_summary,
    setup_dtd_symlink,
    setup_visa_pytorch,
    summarize_synthetic_split,
    to_numpy,
)
from src.anomaly_detection.patchcore.config import DEFAULT_CONFIG_PATH, load_experiment_config
from src.anomaly_detection.patchcore.model import build_patchcore_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchCore on VisA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def get_categories(config) -> list[str]:
    return config.categories or list_visa_categories()


def build_datamodule(config, *, category: str, image_size: int, val_split_ratio: float):
    return build_visa_synthetic_datamodule(
        dataset_root=config.dataset_root,
        category=category,
        image_size=image_size,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        val_split_ratio=val_split_ratio,
        seed=config.train.seed,
    )


def run_single(config, *, output_dir: Path, backbone: str, image_size: int, coreset_sampling_ratio: float) -> dict:
    from anomalib.engine import Engine

    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    category_frames: list[pd.DataFrame] = []
    pixel_rows: list[dict] = []
    train_split_rows: list[dict] = []

    for category in get_categories(config):
        datamodule = build_datamodule(config, category=category, image_size=image_size, val_split_ratio=config.train.val_split_ratio)
        model = build_patchcore_model(
            backbone=backbone,
            layers=config.model.layers,
            pre_trained=config.model.pre_trained,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=config.model.num_neighbors,
            image_size=image_size,
        )
        run_dir = output_dir / category / "anomalib_run"
        engine = Engine(default_root_dir=str(run_dir), accelerator=accelerator, devices=1)
        engine.fit(model=model, datamodule=datamodule)
        train_split_rows.append(summarize_synthetic_split(datamodule, category=category).__dict__)

        ckpt = _resolve_checkpoint(engine, run_dir)
        predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt, return_predictions=True)
        pred_df, anomaly_maps, masks = _parse_predictions(predictions, category=category)
        category_frames.append(pred_df)
        pixel_rows.append({"category": category, **compute_pixel_level_metrics(anomaly_maps, masks)})

    return _save_run_results(
        output_dir,
        category_frames,
        pixel_rows,
        train_split_rows=train_split_rows,
        backbone=backbone,
        image_size=image_size,
        coreset_sampling_ratio=coreset_sampling_ratio,
    )


def run_test(config, best_artifact_path: Path, *, model_params: dict | None = None) -> dict:
    from anomalib.engine import Engine

    output_dir = config.output_path / "best_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    category_frames: list[pd.DataFrame] = []
    pixel_rows: list[dict] = []
    train_split_rows: list[dict] = []
    params = model_params or {}
    backbone = params.get("backbone", config.model.backbone)
    image_size = int(params.get("image_size", config.train.image_size))
    coreset_sampling_ratio = float(params.get("coreset_sampling_ratio", config.model.coreset_sampling_ratio))

    for category in get_categories(config):
        datamodule = build_datamodule(config, category=category, image_size=image_size, val_split_ratio=config.train.val_split_ratio)
        model = build_patchcore_model(
            backbone=backbone,
            layers=config.model.layers,
            pre_trained=config.model.pre_trained,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=config.model.num_neighbors,
            image_size=image_size,
        )
        ckpt = str(_find_category_checkpoint(best_artifact_path, category))
        run_dir = output_dir / category / "anomalib_run"
        engine = Engine(default_root_dir=str(run_dir), accelerator=accelerator, devices=1)
        datamodule.setup()
        train_split_rows.append(summarize_synthetic_split(datamodule, category=category).__dict__)
        predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt, return_predictions=True)
        pred_df, anomaly_maps, masks = _parse_predictions(predictions, category=category)
        category_frames.append(pred_df)
        pixel_rows.append({"category": category, **compute_pixel_level_metrics(anomaly_maps, masks)})

    return _save_run_results(output_dir, category_frames, pixel_rows, train_split_rows=train_split_rows, **params)


def run_grid_tuning(config) -> pd.DataFrame:
    tuning_root = config.output_path / "grid"
    rows = []
    for backbone in config.grid.backbones:
        for image_size in config.grid.image_sizes:
            for coreset_ratio in config.grid.coreset_sampling_ratios:
                trial_name = f"{backbone}_sz{image_size}_cr{coreset_ratio}"
                result = run_single(config, output_dir=tuning_root / trial_name, backbone=backbone, image_size=image_size, coreset_sampling_ratio=coreset_ratio)
                rows.append({"trial_name": trial_name, **result})

    results_df = pd.DataFrame(rows).sort_values("image_auroc", ascending=False).reset_index(drop=True)
    save_dataframe(tuning_root / "tuning_results.csv", results_df)
    return results_df


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    setup_visa_pytorch(config.dataset_root)
    setup_dtd_symlink(dtd_data_root=Path(config.dataset_root).parent, project_root=PROJECT_ROOT)

    if config.tuning_mode == "grid":
        results_df = run_grid_tuning(config)
        best = results_df.iloc[0].to_dict()
        summary = {
            "experiment_name": config.experiment_name,
            "mode": "grid",
            "best_source": "grid",
            "best_artifact_path": str(config.output_path / "grid" / best["trial_name"]),
            "best_model_config": {
                "backbone": best["backbone"],
                "image_size": int(best["image_size"]),
                "coreset_sampling_ratio": float(best["coreset_sampling_ratio"]),
            },
            "best_image_level_metric": float(best["image_auroc"]),
            "best_pixel_level_metric": float(best["pixel_auroc"]) if not pd.isna(best.get("pixel_auroc", float("nan"))) else float("nan"),
            "test_image_level_metric": None,
            "test_pixel_level_metric": None,
        }
    elif config.tuning_mode == "none":
        result = run_single(config, output_dir=config.output_path / "train", backbone=config.model.backbone, image_size=config.train.image_size, coreset_sampling_ratio=config.model.coreset_sampling_ratio)
        summary = {
            "experiment_name": config.experiment_name,
            "mode": "none",
            "best_source": "train",
            "best_artifact_path": str(config.output_path / "train"),
            "best_model_config": {
                "backbone": config.model.backbone,
                "image_size": int(config.train.image_size),
                "coreset_sampling_ratio": float(config.model.coreset_sampling_ratio),
            },
            "best_image_level_metric": float(result["image_auroc"]),
            "best_pixel_level_metric": float(result["pixel_auroc"]) if not pd.isna(result.get("pixel_auroc", float("nan"))) else float("nan"),
            "test_image_level_metric": None,
            "test_pixel_level_metric": None,
        }
    else:
        raise ValueError(f"unsupported tuning_mode: {config.tuning_mode}")

    save_json(config.output_path / "experiment_summary.json", summary)
    print("training completed.")
    print(f"best artifact: {summary['best_artifact_path']}")


# ── helpers (same as padim/train.py) ─────────────────────────────────────────

def _resolve_checkpoint(engine, run_dir: Path) -> str:
    trainer = getattr(engine, "trainer", None)
    cb = getattr(trainer, "checkpoint_callback", None)
    best = getattr(cb, "best_model_path", "") if cb else ""
    if best:
        return best
    ckpts = sorted(run_dir.rglob("*.ckpt"))
    if ckpts:
        return str(ckpts[-1])
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def _find_category_checkpoint(artifact_path: Path, category: str) -> Path:
    search_dir = artifact_path / category / "anomalib_run"
    ckpts = sorted(search_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {search_dir}")
    return ckpts[-1]


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


def _save_run_results(output_dir: Path, category_frames: list, pixel_rows: list, train_split_rows: list[dict] | None = None, **extra) -> dict:
    from src.anomaly_detection.common import compute_image_level_metrics

    predictions_df = pd.concat(category_frames, ignore_index=True)
    image_metrics = compute_image_level_metrics(predictions_df)
    train_size_df = pd.DataFrame(train_split_rows or [])
    category_metrics_df = build_category_metrics(predictions_df, train_size_df=train_size_df)
    pixel_df = pd.DataFrame(pixel_rows)
    valid_pixel = pixel_df.dropna(subset=["pixel_auroc"])
    pixel_auroc = float(valid_pixel["pixel_auroc"].mean()) if len(valid_pixel) else float("nan")
    pixel_ap = float(valid_pixel["pixel_ap"].mean()) if len(valid_pixel) else float("nan")
    threshold_curve_df = build_threshold_curve(predictions_df)

    save_dataframe(output_dir / "predictions.csv", predictions_df)
    save_dataframe(output_dir / "category_metrics.csv", category_metrics_df)
    if len(train_size_df):
        save_dataframe(output_dir / "train_split_summary.csv", train_size_df)
    save_dataframe(output_dir / "threshold_curve.csv", threshold_curve_df)
    save_json(output_dir / "image_level_metrics.json", image_metrics)
    pixel_metrics = {"pixel_auroc": pixel_auroc, "pixel_ap": pixel_ap}
    save_json(output_dir / "pixel_level_metrics.json", pixel_metrics)
    save_result_summary(output_dir, image_metrics=image_metrics, pixel_metrics=pixel_metrics, extra=extra)

    return {"output_dir": str(output_dir), "image_auroc": float(image_metrics["image_auroc"]), "pixel_auroc": pixel_auroc, **extra}


if __name__ == "__main__":
    main()
