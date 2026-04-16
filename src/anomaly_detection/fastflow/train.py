from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.common import (
    build_visa_datamodule,
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
    to_numpy,
)
from src.anomaly_detection.fastflow.config import DEFAULT_CONFIG_PATH, load_experiment_config
from src.anomaly_detection.fastflow.model import build_fastflow_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FastFlow on VisA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def get_categories(config) -> list[str]:
    return config.categories or list_visa_categories()


def build_datamodule(config, *, category: str, image_size: int):
    return build_visa_datamodule(
        dataset_root=config.dataset_root,
        category=category,
        train_batch_size=config.train.batch_size,
        eval_batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        val_split_mode="synthetic",
        val_split_ratio=config.train.val_split_ratio,
        seed=config.train.seed,
    )


def run_single(
    config,
    *,
    output_dir: Path,
    backbone: str,
    image_size: int,
    flow_steps: int,
    hidden_ratio: float,
    optuna_trial: optuna.Trial | None = None,
) -> dict:
    from anomalib.engine import Engine
    from optuna_integration import PyTorchLightningPruningCallback

    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    category_frames: list[pd.DataFrame] = []
    pixel_rows: list[dict] = []
    train_losses: list[float] = []

    for category in get_categories(config):
        datamodule = build_datamodule(config, category=category, image_size=image_size)
        model = build_fastflow_model(backbone=backbone, pre_trained=config.model.pre_trained, flow_steps=flow_steps, hidden_ratio=hidden_ratio, image_size=image_size)
        run_dir = output_dir / category / "anomalib_run"

        callbacks = []
        if optuna_trial is not None:
            callbacks.append(PyTorchLightningPruningCallback(optuna_trial, monitor="train_loss"))

        engine = Engine(
            default_root_dir=str(run_dir),
            accelerator=accelerator,
            devices=1,
            max_epochs=config.train.max_epochs,
            callbacks=callbacks,
        )
        engine.fit(model=model, datamodule=datamodule)
        train_losses.append(_extract_train_loss(engine))

        ckpt = _resolve_checkpoint(engine, run_dir)
        predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt, return_predictions=True)
        pred_df, anomaly_maps, masks = _parse_predictions(predictions, category=category)
        category_frames.append(pred_df)
        pixel_rows.append({"category": category, **compute_pixel_level_metrics(anomaly_maps, masks)})

    return _save_run_results(
        output_dir,
        category_frames,
        pixel_rows,
        backbone=backbone,
        image_size=image_size,
        flow_steps=flow_steps,
        hidden_ratio=hidden_ratio,
        train_loss=float(sum(train_losses) / len(train_losses)),
    )


def run_test(config, best_artifact_path: Path, *, model_params: dict | None = None) -> dict:
    from anomalib.engine import Engine

    output_dir = config.output_path / "best_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    category_frames: list[pd.DataFrame] = []
    pixel_rows: list[dict] = []
    params = model_params or {}
    backbone = params.get("backbone", config.model.backbone)
    image_size = int(params.get("image_size", config.train.image_size))
    flow_steps = int(params.get("flow_steps", config.model.flow_steps))
    hidden_ratio = float(params.get("hidden_ratio", config.model.hidden_ratio))

    for category in get_categories(config):
        datamodule = build_datamodule(config, category=category, image_size=image_size)
        model = build_fastflow_model(
            backbone=backbone,
            pre_trained=config.model.pre_trained,
            flow_steps=flow_steps,
            hidden_ratio=hidden_ratio,
            image_size=image_size,
        )
        ckpt = str(_find_category_checkpoint(best_artifact_path, category))
        run_dir = output_dir / category / "anomalib_run"
        engine = Engine(default_root_dir=str(run_dir), accelerator=accelerator, devices=1)
        predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt, return_predictions=True)
        pred_df, anomaly_maps, masks = _parse_predictions(predictions, category=category)
        category_frames.append(pred_df)
        pixel_rows.append({"category": category, **compute_pixel_level_metrics(anomaly_maps, masks)})

    return _save_run_results(output_dir, category_frames, pixel_rows, **params)


def run_grid_tuning(config) -> pd.DataFrame:
    tuning_root = config.output_path / "grid"
    rows = []
    for backbone in config.grid.backbones:
        for image_size in config.grid.image_sizes:
            for flow_steps in config.grid.flow_steps_list:
                trial_name = f"{backbone}_sz{image_size}_fs{flow_steps}"
                result = run_single(config, output_dir=tuning_root / trial_name, backbone=backbone, image_size=image_size, flow_steps=flow_steps, hidden_ratio=config.model.hidden_ratio)
                rows.append({"trial_name": trial_name, **result})

    results_df = pd.DataFrame(rows).sort_values("train_loss", ascending=True).reset_index(drop=True)
    save_dataframe(tuning_root / "tuning_results.csv", results_df)
    return results_df


def run_optuna_tuning(config) -> pd.DataFrame:
    grid_df = pd.read_csv(config.output_path / "grid" / "tuning_results.csv")
    best_grid = grid_df.iloc[0].to_dict()

    backbone_candidates = config.optuna.backbone_candidates or config.grid.backbones
    image_size_candidates = config.optuna.image_size_candidates or [int(best_grid["image_size"])]
    flow_steps_candidates = config.optuna.flow_steps_candidates or sorted({max(4, int(best_grid["flow_steps"]) - 4), int(best_grid["flow_steps"]), int(best_grid["flow_steps"]) + 4})
    hidden_ratio_candidates = config.optuna.hidden_ratio_candidates or [0.5, 1.0, 2.0]

    output_dir = config.output_path / "optuna"
    rows = []

    def objective(trial: optuna.Trial) -> float:
        backbone = trial.suggest_categorical("backbone", backbone_candidates)
        image_size = trial.suggest_categorical("image_size", image_size_candidates)
        flow_steps = trial.suggest_categorical("flow_steps", flow_steps_candidates)
        hidden_ratio = trial.suggest_categorical("hidden_ratio", hidden_ratio_candidates)
        trial_name = f"trial_{trial.number:03d}"
        try:
            result = run_single(config, output_dir=output_dir / trial_name, backbone=backbone, image_size=image_size, flow_steps=flow_steps, hidden_ratio=hidden_ratio, optuna_trial=trial)
        except optuna.TrialPruned:
            raise
        rows.append({"trial_name": trial_name, **result})
        return float(result["train_loss"])

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name=f"{config.experiment_name}_optuna")
    study.optimize(objective, n_trials=config.optuna.n_trials)

    results_df = pd.DataFrame(rows).sort_values("train_loss", ascending=True).reset_index(drop=True)
    save_dataframe(output_dir / "optuna_results.csv", results_df)
    save_json(output_dir / "optuna_summary.json", {"best_params": study.best_params, "best_value": study.best_value})
    return results_df


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    setup_visa_pytorch(config.dataset_root)
    setup_dtd_symlink(dtd_data_root=Path(config.dataset_root).parent, project_root=PROJECT_ROOT)

    if config.tuning_mode == "full":
        grid_df = run_grid_tuning(config)
        optuna_df = run_optuna_tuning(config)
        best = optuna_df.iloc[0].to_dict()
        summary = {
            "experiment_name": config.experiment_name,
            "mode": "full",
            "best_source": "optuna",
            "best_artifact_path": str(config.output_path / "optuna" / best["trial_name"]),
            "best_model_config": {
                "backbone": best["backbone"],
                "image_size": int(best["image_size"]),
                "flow_steps": int(best["flow_steps"]),
                "hidden_ratio": float(best["hidden_ratio"]),
            },
            "best_image_level_metric": float(best["image_auroc"]),
            "best_pixel_level_metric": float(best["pixel_auroc"]) if not pd.isna(best.get("pixel_auroc", float("nan"))) else float("nan"),
            "test_image_level_metric": None,
            "test_pixel_level_metric": None,
        }
    elif config.tuning_mode == "grid":
        grid_df = run_grid_tuning(config)
        best = grid_df.iloc[0].to_dict()
        summary = {
            "experiment_name": config.experiment_name,
            "mode": "grid",
            "best_source": "grid",
            "best_artifact_path": str(config.output_path / "grid" / best["trial_name"]),
            "best_model_config": {
                "backbone": best["backbone"],
                "image_size": int(best["image_size"]),
                "flow_steps": int(best["flow_steps"]),
                "hidden_ratio": float(best["hidden_ratio"]),
            },
            "best_image_level_metric": float(best["image_auroc"]),
            "best_pixel_level_metric": float(best["pixel_auroc"]) if not pd.isna(best.get("pixel_auroc", float("nan"))) else float("nan"),
            "test_image_level_metric": None,
            "test_pixel_level_metric": None,
        }
    elif config.tuning_mode == "none":
        result = run_single(config, output_dir=config.output_path / "train", backbone=config.model.backbone, image_size=config.train.image_size, flow_steps=config.model.flow_steps, hidden_ratio=config.model.hidden_ratio)
        summary = {
            "experiment_name": config.experiment_name,
            "mode": "none",
            "best_source": "train",
            "best_artifact_path": str(config.output_path / "train"),
            "best_model_config": {
                "backbone": config.model.backbone,
                "image_size": int(config.train.image_size),
                "flow_steps": int(config.model.flow_steps),
                "hidden_ratio": float(config.model.hidden_ratio),
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


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _extract_train_loss(engine) -> float:
    trainer = getattr(engine, "trainer", None)
    if trainer is None:
        raise RuntimeError("trainer not available after fit")
    metrics = getattr(trainer, "callback_metrics", {}) or {}
    train_loss = metrics.get("train_loss")
    if train_loss is None:
        metrics = getattr(trainer, "logged_metrics", {}) or {}
        train_loss = metrics.get("train_loss")
    if train_loss is None:
        raise RuntimeError("train_loss was not logged by the trainer")
    return float(train_loss.detach().cpu().item() if hasattr(train_loss, "detach") else train_loss)


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


def _save_run_results(output_dir: Path, category_frames: list, pixel_rows: list, **extra) -> dict:
    from src.anomaly_detection.common import compute_image_level_metrics

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
    save_result_summary(output_dir, image_metrics=image_metrics, pixel_metrics=pixel_metrics, extra=extra)

    return {"output_dir": str(output_dir), "image_auroc": float(image_metrics["image_auroc"]), "pixel_auroc": pixel_auroc, **extra}


if __name__ == "__main__":
    main()
