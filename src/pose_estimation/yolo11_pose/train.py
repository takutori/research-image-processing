from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.yolo11_pose.config import DEFAULT_CONFIG_PATH, YOLO11PoseExperimentConfig, load_experiment_config
from src.pose_estimation.yolo11_pose.model import build_yolo11_pose_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11-pose on COCO person keypoints.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def run_single_training(
    config: YOLO11PoseExperimentConfig,
    *,
    output_dir: Path,
    model_name: str,
    img_size: int,
    epochs: int,
    patience: int,
    batch: int,
    workers: int,
    fraction: float,
    device: str,
    run_name: str,
) -> dict[str, object]:
    model = build_yolo11_pose_model(model_name)
    try:
        results = model.train(
            data=config.dataset_yaml,
            project=str(output_dir),
            name=run_name,
            imgsz=img_size,
            epochs=epochs,
            patience=patience,
            batch=batch,
            workers=workers,
            fraction=fraction,
            device=device,
            exist_ok=True,
            verbose=False,
        )
    except Exception as exc:
        message = str(exc)
        if "CUDA error" in message or "cudaErrorUnknown" in message:
            raise RuntimeError(
                "YOLO11-pose training failed with a CUDA runtime error. "
                "First try a lighter config: tuning_mode=none, model=yolo11n-pose.pt, "
                "batch=8, workers=4. If it still fails, re-run once with "
                "CUDA_LAUNCH_BLOCKING=1 to distinguish a config issue from a host/driver reset."
            ) from exc
        raise
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    best_ap50 = float(metrics.get("metrics/mAP50(P)", 0.0))
    best_ap = float(metrics.get("metrics/mAP50-95(P)", 0.0))
    return {
        "run_name": run_name,
        "best_checkpoint_path": str(output_dir / run_name / "weights" / "best.pt"),
        "best_valid_ap": best_ap,
        "best_valid_ap50": best_ap50,
        "model_name": model_name,
        "img_size": img_size,
    }


def run_grid_tuning(config: YOLO11PoseExperimentConfig, device: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in config.grid.model_names:
        for img_size in config.grid.img_sizes:
            run_name = f"{model_name.replace('.pt', '').replace('/', '_')}_sz{img_size}"
            rows.append(
                run_single_training(
                    config,
                    output_dir=config.output_path / "grid",
                    model_name=model_name,
                    img_size=img_size,
                    epochs=config.grid.epochs,
                    patience=config.grid.patience,
                    batch=config.grid.batch,
                    workers=config.grid.workers,
                    fraction=config.grid.fraction,
                    device=device,
                    run_name=run_name,
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = args.device or config.train.device
    config.output_path.mkdir(parents=True, exist_ok=True)

    if config.tuning_mode == "grid":
        rows = run_grid_tuning(config, device)
        best = max(rows, key=lambda row: float(row["best_valid_ap"]))
        best_source = "grid"
        mode = "grid"
    else:
        best = run_single_training(
            config,
            output_dir=config.output_path,
            model_name=config.train.model_name,
            img_size=config.train.img_size,
            epochs=config.train.epochs,
            patience=config.train.patience,
            batch=config.train.batch,
            workers=config.train.workers,
            fraction=config.train.fraction,
            device=device,
            run_name="train",
        )
        best_source = "train"
        mode = "none"

    summary = {
        "experiment_name": config.experiment_name,
        "mode": mode,
        "best_source": best_source,
        "best_run_name": best["run_name"],
        "best_checkpoint_path": best["best_checkpoint_path"],
        "best_valid_ap": best["best_valid_ap"],
        "best_valid_ap50": best["best_valid_ap50"],
        "best_model_config": {
            "model_name": best["model_name"],
            "img_size": best["img_size"],
        },
        "test_ap": None,
        "test_ap50": None,
    }
    (config.output_path / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"experiment_summary.json saved to: {config.output_path / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
