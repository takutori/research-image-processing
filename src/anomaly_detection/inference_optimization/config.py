from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "anomaly_detection" / "inference_optimization_visa.yaml"

SUPPORTED_MODELS = ("padim", "patchcore", "fastflow", "winclip_zeroshot")
SUPPORTED_VARIANTS = (
    "baseline_eager_fp32",
    "eager_amp_fp16",
    "torch_compile_fp32",
    "torch_compile_amp_fp16",
    "onnxruntime_cuda_fp32",
    "onnxruntime_tensorrt_fp32",
    "onnxruntime_tensorrt_fp16",
)


@dataclass
class BenchmarkConfig:
    warmup_runs: int = 1
    repeat_runs: int = 5
    max_batches: int | None = None
    include_pixel_metrics: bool = True
    save_predictions: bool = True
    torch_compile_mode: str = "reduce-overhead"


@dataclass
class ArtifactConfig:
    padim_summary: str = "/workspace/checkpoints/anomaly_detection/padim/padim_visa/experiment_summary.json"
    patchcore_summary: str = "/workspace/checkpoints/anomaly_detection/patchcore/patchcore_visa/experiment_summary.json"
    fastflow_summary: str = "/workspace/checkpoints/anomaly_detection/fastflow/fastflow_visa/experiment_summary.json"
    winclip_zeroshot_summary: str = "/workspace/checkpoints/anomaly_detection/winclip_zeroshot/visa/experiment_summary.json"

    def summary_path(self, model_name: str) -> Path:
        try:
            return Path(getattr(self, f"{model_name}_summary"))
        except AttributeError as exc:
            raise ValueError(f"unsupported model for artifact lookup: {model_name}") from exc


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    categories: list[str] = field(default_factory=lambda: ["candle"])


@dataclass
class InferenceOptimizationConfig:
    experiment_name: str = "inference_optimization_visa"
    dataset_root: str = "/workspace/data/visa"
    output_dir: str = "/workspace/checkpoints/anomaly_detection/inference_optimization/visa"
    models: list[str] = field(default_factory=lambda: list(SUPPORTED_MODELS))
    variants: list[str] = field(default_factory=lambda: list(SUPPORTED_VARIANTS))
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> InferenceOptimizationConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = InferenceOptimizationConfig()
    cfg.experiment_name = raw.get("experiment_name", cfg.experiment_name)
    cfg.dataset_root = raw.get("dataset_root", cfg.dataset_root)
    cfg.output_dir = raw.get("output_dir", cfg.output_dir)
    cfg.models = list(raw.get("models", cfg.models))
    cfg.variants = list(raw.get("variants", cfg.variants))

    if "benchmark" in raw:
        data = raw["benchmark"]
        cfg.benchmark.warmup_runs = int(data.get("warmup_runs", cfg.benchmark.warmup_runs))
        cfg.benchmark.repeat_runs = int(data.get("repeat_runs", cfg.benchmark.repeat_runs))
        max_batches = data.get("max_batches", cfg.benchmark.max_batches)
        cfg.benchmark.max_batches = None if max_batches is None else int(max_batches)
        cfg.benchmark.include_pixel_metrics = bool(data.get("include_pixel_metrics", cfg.benchmark.include_pixel_metrics))
        cfg.benchmark.save_predictions = bool(data.get("save_predictions", cfg.benchmark.save_predictions))
        cfg.benchmark.torch_compile_mode = data.get("torch_compile_mode", cfg.benchmark.torch_compile_mode)

    if "artifacts" in raw:
        data = raw["artifacts"]
        cfg.artifacts.padim_summary = data.get("padim_summary", cfg.artifacts.padim_summary)
        cfg.artifacts.patchcore_summary = data.get("patchcore_summary", cfg.artifacts.patchcore_summary)
        cfg.artifacts.fastflow_summary = data.get("fastflow_summary", cfg.artifacts.fastflow_summary)
        cfg.artifacts.winclip_zeroshot_summary = data.get("winclip_zeroshot_summary", cfg.artifacts.winclip_zeroshot_summary)

    if "runtime" in raw:
        data = raw["runtime"]
        cfg.runtime.device = data.get("device", cfg.runtime.device)
        cfg.runtime.categories = list(data.get("categories", cfg.runtime.categories))

    _validate_config(cfg)
    return cfg


def _validate_config(config: InferenceOptimizationConfig) -> None:
    unknown_models = sorted(set(config.models) - set(SUPPORTED_MODELS))
    if unknown_models:
        raise ValueError(f"unsupported models: {unknown_models}. supported={list(SUPPORTED_MODELS)}")
    unknown_variants = sorted(set(config.variants) - set(SUPPORTED_VARIANTS))
    if unknown_variants:
        raise ValueError(f"unsupported variants: {unknown_variants}. supported={list(SUPPORTED_VARIANTS)}")
    if config.benchmark.warmup_runs < 0:
        raise ValueError("benchmark.warmup_runs must be >= 0")
    if config.benchmark.repeat_runs <= 0:
        raise ValueError("benchmark.repeat_runs must be > 0")
    if config.benchmark.max_batches is not None and config.benchmark.max_batches <= 0:
        raise ValueError("benchmark.max_batches must be null or > 0")
