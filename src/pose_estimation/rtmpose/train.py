from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.rtmpose.config import DEFAULT_CONFIG_PATH, RTMPoseExperimentConfig, load_experiment_config
from src.pose_estimation.rtmpose.model import resolve_mmpose_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RTMPose with config-driven MMPose wrapper.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def _require_mmengine() -> tuple:
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RTMPose training requires mmengine and mmpose at runtime."
        ) from exc
    return Config, Runner


def _unwrap_dataset_cfg(dataset_cfg):
    current = dataset_cfg
    while isinstance(current, dict) and "dataset" in current:
        current = current["dataset"]
    return current


def _set_dataset_paths(cfg, dataset_root: str) -> None:
    dataset_root_path = Path(dataset_root)
    train_ann = str(dataset_root_path / "annotations" / "person_keypoints_train2017.json")
    val_ann = str(dataset_root_path / "annotations" / "person_keypoints_val2017.json")
    train_prefix = str(dataset_root_path / "images" / "train2017")
    val_prefix = str(dataset_root_path / "images" / "val2017")

    train_dataset = _unwrap_dataset_cfg(cfg.train_dataloader["dataset"])
    train_dataset["data_root"] = dataset_root
    train_dataset["ann_file"] = train_ann
    train_dataset["data_prefix"] = {"img": train_prefix}

    for dataloader_key in ("val_dataloader", "test_dataloader"):
        dataset = _unwrap_dataset_cfg(cfg[dataloader_key]["dataset"])
        dataset["data_root"] = dataset_root
        dataset["ann_file"] = val_ann
        dataset["data_prefix"] = {"img": val_prefix}

    if "val_evaluator" in cfg:
        cfg.val_evaluator["ann_file"] = val_ann
    if "test_evaluator" in cfg:
        cfg.test_evaluator["ann_file"] = val_ann


def _extract_history(work_dir: Path) -> pd.DataFrame:
    scalars_path = work_dir / "vis_data" / "scalars.json"
    if not scalars_path.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for line in scalars_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


def _extract_metrics(history_df: pd.DataFrame) -> tuple[float, float]:
    if history_df.empty:
        return 0.0, 0.0
    ap_candidates = [column for column in history_df.columns if column.endswith("coco/AP")]
    ap50_candidates = [column for column in history_df.columns if column.endswith("coco/AP .5")]
    best_ap = float(history_df[ap_candidates[0]].max()) if ap_candidates else 0.0
    best_ap50 = float(history_df[ap50_candidates[0]].max()) if ap50_candidates else 0.0
    return best_ap, best_ap50


def _find_best_checkpoint(work_dir: Path) -> Path:
    candidates = sorted(work_dir.rglob("best*.pth"))
    if candidates:
        return candidates[0]
    latest_candidates = sorted(work_dir.rglob("latest.pth"))
    if latest_candidates:
        return latest_candidates[0]
    generic_candidates = sorted(work_dir.rglob("*.pth"))
    if not generic_candidates:
        raise FileNotFoundError(f"no checkpoint found under {work_dir}")
    return generic_candidates[0]


def run_single_training(
    config: RTMPoseExperimentConfig,
    *,
    output_dir: Path,
    config_path: str,
    checkpoint: str | None,
    learning_rate: float,
    max_epochs: int,
    run_name: str,
) -> dict[str, object]:
    Config, Runner = _require_mmengine()
    work_dir = output_dir / run_name
    work_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = resolve_mmpose_config_path(config_path)
    if not Path(resolved_config_path).exists():
        raise FileNotFoundError(
            f"RTMPose config not found: {config_path}. "
            "If you installed mmpose via pip, make sure the config exists under the installed package."
        )
    cfg = Config.fromfile(resolved_config_path)
    cfg.work_dir = str(work_dir)
    if checkpoint:
        cfg.load_from = checkpoint
    _set_dataset_paths(cfg, config.dataset_root)
    if "train_cfg" in cfg and isinstance(cfg.train_cfg, dict):
        cfg.train_cfg["max_epochs"] = max_epochs
        cfg.train_cfg["val_interval"] = min(config.train.val_interval, max_epochs)
    if "optim_wrapper" in cfg and "optimizer" in cfg.optim_wrapper:
        cfg.optim_wrapper["optimizer"]["lr"] = learning_rate

    runner = Runner.from_cfg(cfg)
    runner.train()

    history_df = _extract_history(work_dir)
    if not history_df.empty:
        history_df.to_csv(work_dir / "history.csv", index=False)
    best_ap, best_ap50 = _extract_metrics(history_df)
    best_checkpoint_path = _find_best_checkpoint(work_dir)
    return {
        "run_name": run_name,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_valid_ap": best_ap,
        "best_valid_ap50": best_ap50,
        "config_path": resolved_config_path,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
    }


def run_grid_tuning(config: RTMPoseExperimentConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config_path in config.grid.config_paths:
        for learning_rate in config.grid.learning_rates:
            config_slug = Path(config_path).stem.replace(".", "_")
            lr_slug = str(learning_rate).replace(".", "p")
            rows.append(
                run_single_training(
                    config,
                    output_dir=config.output_path / "grid",
                    config_path=config_path,
                    checkpoint=config.train.checkpoint,
                    learning_rate=learning_rate,
                    max_epochs=config.grid.max_epochs,
                    run_name=f"{config_slug}_lr{lr_slug}",
                )
            )
    return rows


def run_optuna_tuning(config: RTMPoseExperimentConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        config_path = trial.suggest_categorical("config_path", config.grid.config_paths)
        learning_rate = trial.suggest_float(
            "learning_rate",
            config.optuna.learning_rate_low,
            config.optuna.learning_rate_high,
            log=True,
        )
        result = run_single_training(
            config,
            output_dir=config.output_path / "optuna",
            config_path=config_path,
            checkpoint=config.train.checkpoint,
            learning_rate=learning_rate,
            max_epochs=config.optuna.max_epochs,
            run_name=f"trial_{trial.number:03d}",
        )
        rows.append(result)
        return float(result["best_valid_ap"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.optuna.n_trials)
    pd.DataFrame(rows).to_csv(config.output_path / "optuna" / "optuna_results.csv", index=False)
    (config.output_path / "optuna" / "optuna_summary.json").write_text(
        json.dumps({"best_params": study.best_params, "best_value": study.best_value}, ensure_ascii=False, indent=2)
    )
    return rows


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    config.output_path.mkdir(parents=True, exist_ok=True)

    if config.tuning_mode == "full":
        grid_rows = run_grid_tuning(config)
        pd.DataFrame(grid_rows).to_csv(config.output_path / "grid" / "tuning_results.csv", index=False)
        optuna_rows = run_optuna_tuning(config)
        best = max(optuna_rows, key=lambda row: float(row["best_valid_ap"]))
        mode = "full"
        best_source = "optuna"
    elif config.tuning_mode == "grid":
        rows = run_grid_tuning(config)
        pd.DataFrame(rows).to_csv(config.output_path / "grid" / "tuning_results.csv", index=False)
        best = max(rows, key=lambda row: float(row["best_valid_ap"]))
        mode = "grid"
        best_source = "grid"
    else:
        best = run_single_training(
            config,
            output_dir=config.output_path,
            config_path=config.train.config_path,
            checkpoint=config.train.checkpoint,
            learning_rate=config.train.learning_rate,
            max_epochs=config.train.max_epochs,
            run_name="train",
        )
        mode = "none"
        best_source = "train"

    summary = {
        "experiment_name": config.experiment_name,
        "mode": mode,
        "best_source": best_source,
        "best_run_name": best["run_name"],
        "best_checkpoint_path": best["best_checkpoint_path"],
        "best_valid_ap": best["best_valid_ap"],
        "best_valid_ap50": best["best_valid_ap50"],
        "best_model_config": {
            "config_path": best["config_path"],
            "learning_rate": best["learning_rate"],
            "max_epochs": best["max_epochs"],
        },
        "test_ap": None,
        "test_ap50": None,
    }
    (config.output_path / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"experiment_summary.json saved to: {config.output_path / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
