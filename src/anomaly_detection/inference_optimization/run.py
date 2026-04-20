from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.inference_optimization.artifacts import load_model_artifact, write_json
from src.anomaly_detection.inference_optimization.backends import build_backend
from src.anomaly_detection.inference_optimization.config import DEFAULT_CONFIG_PATH, load_config
from src.anomaly_detection.inference_optimization.data import PreparedBatch, prepare_test_batches
from src.anomaly_detection.inference_optimization.metrics import build_prediction_output, summarize_predictions
from src.anomaly_detection.inference_optimization.models import build_model_and_spec, config_path_for_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark post-training inference optimization for anomaly detection models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of models from config.")
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of variants from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    models = args.models or config.models
    variants = args.variants or config.variants
    device = _resolve_device(config.runtime.device)
    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_run_files(output_dir)
    write_json(output_dir / "environment.json", _collect_environment(device=device))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for model_name in models:
        artifact = load_model_artifact(model_name, config.artifacts.summary_path(model_name))
        for category in config.runtime.categories:
            print(f"benchmarking {model_name}/{category}")
            try:
                initial_model, spec = build_model_and_spec(
                    model_name=model_name,
                    artifact=artifact,
                    category=category,
                    config_path=config_path_for_model(model_name),
                    device=device,
                )
                del initial_model
                batches = prepare_test_batches(
                    dataset_root=config.dataset_root,
                    categories=[category],
                    image_size=spec.image_size,
                    batch_size=spec.batch_size,
                    num_workers=spec.num_workers,
                    seed=spec.seed,
                    device=device,
                    max_batches=config.benchmark.max_batches,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(_error_row(model_name=model_name, category=category, variant="prepare", exc=exc))
                _flush_progress(output_dir=output_dir, rows=rows, errors=errors)
                continue

            for variant in variants:
                row, error = _run_variant(
                    model_name=model_name,
                    category=category,
                    variant=variant,
                    artifact=artifact,
                    config_path=config_path_for_model(model_name),
                    device=device,
                    batches=batches,
                    output_dir=output_dir,
                    config=config,
                )
                if row:
                    rows.append(row)
                if error:
                    errors.append(error)
                _flush_progress(output_dir=output_dir, rows=rows, errors=errors)

    _flush_progress(output_dir=output_dir, rows=rows, errors=errors)
    print(f"inference optimization benchmark files written to: {output_dir}")


def _run_variant(
    *,
    model_name: str,
    category: str,
    variant: str,
    artifact,
    config_path: Path,
    device: torch.device,
    batches: list[PreparedBatch],
    output_dir: Path,
    config,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        model, _ = build_model_and_spec(
            model_name=model_name,
            artifact=artifact,
            category=category,
            config_path=config_path,
            device=device,
        )
        sample_input = batches[0].images
        backend = build_backend(
            variant=variant,
            model=model,
            device=device,
            sample_input=sample_input,
            export_dir=output_dir / "exports" / model_name / category,
            torch_compile_mode=config.benchmark.torch_compile_mode,
        )
        result = backend.run(batches, warmup_runs=config.benchmark.warmup_runs, repeat_runs=config.benchmark.repeat_runs)
        prediction_output = _build_outputs(batches=batches, pred_scores=result.pred_scores, anomaly_maps=result.anomaly_maps)
        metrics = summarize_predictions(prediction_output)
        if config.benchmark.save_predictions:
            pred_dir = output_dir / "predictions" / model_name / category
            pred_dir.mkdir(parents=True, exist_ok=True)
            prediction_output.predictions.to_csv(pred_dir / f"{variant}.csv", index=False)
        metric_dir = output_dir / "metrics" / model_name / category
        write_json(metric_dir / f"{variant}.json", metrics)

        total_images = int(sum(batch.batch_size for batch in batches))
        return (
            {
                "model": model_name,
                "category": category,
                "variant": variant,
                "status": "ok",
                "total_images": total_images,
                "mean_inference_sec": result.mean_sec,
                "std_inference_sec": result.std_sec,
                "images_per_sec": total_images / result.mean_sec if result.mean_sec > 0 else float("nan"),
                "timings_sec": json.dumps(result.timings_sec),
                **metrics,
            },
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return None, _error_row(model_name=model_name, category=category, variant=variant, exc=exc)


def _build_outputs(
    *,
    batches: list[PreparedBatch],
    pred_scores: list[np.ndarray],
    anomaly_maps: list[np.ndarray],
):
    image_paths: list[str | None] = []
    categories: list[str] = []
    labels: list[np.ndarray] = []
    masks = []
    for batch in batches:
        image_paths.extend(batch.image_paths)
        categories.extend(batch.categories)
        labels.append(batch.labels)
        masks.extend(batch.masks)
    return build_prediction_output(
        image_paths=image_paths,
        categories=categories,
        labels=np.concatenate(labels),
        masks=masks,
        pred_scores=np.concatenate(pred_scores),
        anomaly_maps=np.concatenate(anomaly_maps),
    )


def _add_relative_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speedup_vs_baseline"] = np.nan
    df["best_f1_delta_vs_baseline"] = np.nan
    df["pixel_best_f1_delta_vs_baseline"] = np.nan
    for (model, category), group in df.groupby(["model", "category"]):
        baseline = group[group["variant"] == "baseline_eager_fp32"]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        mask = (df["model"] == model) & (df["category"] == category)
        df.loc[mask, "speedup_vs_baseline"] = float(base["mean_inference_sec"]) / df.loc[mask, "mean_inference_sec"]
        df.loc[mask, "best_f1_delta_vs_baseline"] = df.loc[mask, "best_f1"] - float(base["best_f1"])
        if "pixel_best_f1" in df.columns:
            df.loc[mask, "pixel_best_f1_delta_vs_baseline"] = df.loc[mask, "pixel_best_f1"] - float(base["pixel_best_f1"])
    return df


def _flush_progress(*, output_dir: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    if rows:
        summary_df = _add_relative_columns(pd.DataFrame(rows))
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        write_json(output_dir / "summary.json", {"rows": json.loads(summary_df.to_json(orient="records"))})
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)


def _clear_previous_run_files(output_dir: Path) -> None:
    for filename in ("summary.csv", "summary.json", "errors.csv"):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda but torch.cuda.is_available() is false")
    return torch.device(requested)


def _collect_environment(*, device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ["torch", "anomalib", "onnx", "onnxruntime-gpu", "onnxscript", "tensorrt", "tensorrt-cu12"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(device),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "packages": packages,
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(device)
        env["gpu_capability"] = torch.cuda.get_device_capability(device)
    try:
        import onnxruntime as ort

        env["onnxruntime_available_providers"] = ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001
        env["onnxruntime_available_providers_error"] = repr(exc)
    return env


def _error_row(*, model_name: str, category: str, variant: str, exc: Exception) -> dict[str, str]:
    return {
        "model": model_name,
        "category": category,
        "variant": variant,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


if __name__ == "__main__":
    main()
