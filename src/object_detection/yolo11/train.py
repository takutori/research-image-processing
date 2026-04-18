from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.object_detection.yolo11.config import (
    YOLO11ExperimentConfig,
    load_experiment_config,
)
from src.object_detection.yolo11.model import build_yolo11_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 on COCO.")
    parser.add_argument(
        "--config",
        default="/workspace/config/object_detection/yolo11_coco.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument("--device", default=None, help="Override device (e.g. '0' or 'cpu').")
    return parser.parse_args()


def run_single_training(
    config: YOLO11ExperimentConfig,
    output_dir: Path,
    model_name: str,
    img_size: int,
    epochs: int,
    patience: int,
    batch: int,
    workers: int,
    fraction: float,
    device: str,
    run_name: str = "train",
) -> dict:
    """Train a single YOLO11 run and return result metadata."""
    model = build_yolo11_model(model_name)
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

    best_pt = output_dir / run_name / "weights" / "best.pt"
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    best_map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    best_map = float(metrics.get("metrics/mAP50-95(B)", 0.0))

    return {
        "run_name": run_name,
        "output_dir": str(output_dir / run_name),
        "best_checkpoint_path": str(best_pt),
        "best_valid_map50": best_map50,
        "best_valid_map": best_map,
        "model_name": model_name,
        "img_size": img_size,
        "epochs": epochs,
        "patience": patience,
        "batch": batch,
        "fraction": fraction,
    }


def run_grid_tuning(config: YOLO11ExperimentConfig, device: str) -> list[dict]:
    """Run all grid combinations and return list of result dicts."""
    grid_dir = config.output_path / "grid"
    rows: list[dict] = []

    for model_name in config.grid.model_names:
        for img_size in config.grid.img_sizes:
            slug = model_name.replace(".pt", "").replace("/", "_")
            run_name = f"{slug}_sz{img_size}"
            print(f"\n[grid] {run_name}")
            result = run_single_training(
                config=config,
                output_dir=grid_dir,
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
            rows.append(result)

    return rows


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = args.device or config.train.device

    config.output_path.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_path / "experiment_summary.json"

    if config.tuning_mode == "grid":
        rows = run_grid_tuning(config, device)
        best = max(rows, key=lambda r: r["best_valid_map50"])
        mode = "grid"
        best_source = "grid"
        best_run_name = best["run_name"]
    else:
        result = run_single_training(
            config=config,
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
        best = result
        mode = "none"
        best_source = "train"
        best_run_name = "train"

    summary = {
        "experiment_name": config.experiment_name,
        "mode": mode,
        "best_source": best_source,
        "best_run_name": best_run_name,
        "best_checkpoint_path": best["best_checkpoint_path"],
        "best_valid_map50": best["best_valid_map50"],
        "best_valid_map": best["best_valid_map"],
        "best_model_config": {
            "model_name": best["model_name"],
            "img_size": best["img_size"],
        },
        "test_map50": None,
        "test_map": None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nexperiment_summary.json saved to {summary_path}")
    print(f"best_valid_map50: {best['best_valid_map50']:.4f}")


if __name__ == "__main__":
    main()
