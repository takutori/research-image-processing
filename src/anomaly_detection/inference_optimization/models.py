from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.anomaly_detection.fastflow.config import load_experiment_config as load_fastflow_config
from src.anomaly_detection.fastflow.model import build_fastflow_model
from src.anomaly_detection.inference_optimization.artifacts import ModelArtifact, find_category_checkpoint
from src.anomaly_detection.padim.config import load_experiment_config as load_padim_config
from src.anomaly_detection.padim.model import build_padim_model
from src.anomaly_detection.patchcore.config import load_experiment_config as load_patchcore_config
from src.anomaly_detection.patchcore.model import build_patchcore_model
from src.anomaly_detection.winclip_zeroshot.config import load_experiment_config as load_winclip_config
from src.anomaly_detection.winclip_zeroshot.model import build_winclip_zeroshot_model


@dataclass(frozen=True)
class ModelRuntimeSpec:
    image_size: int
    batch_size: int
    num_workers: int
    seed: int
    categories: list[str]


class PredictionTupleWrapper(torch.nn.Module):
    """Return only tensors needed for speed/metric comparison."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(images)
        pred_score = _get_output_field(output, "pred_score")
        anomaly_map = _get_output_field(output, "anomaly_map")
        if pred_score is None:
            if anomaly_map is None:
                pred_score = torch.zeros(images.shape[0], device=images.device, dtype=images.dtype)
            else:
                pred_score = anomaly_map.flatten(start_dim=1).amax(dim=1)
        if anomaly_map is None:
            anomaly_map = pred_score.reshape(-1, 1, 1)
        return pred_score.reshape(images.shape[0], -1)[:, 0], anomaly_map


def build_model_and_spec(
    *,
    model_name: str,
    artifact: ModelArtifact,
    category: str,
    config_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ModelRuntimeSpec]:
    if model_name == "padim":
        cfg = load_padim_config(config_path)
        params = artifact.best_model_config
        image_size = int(params.get("image_size", cfg.train.image_size))
        model = build_padim_model(
            backbone=params.get("backbone", cfg.model.backbone),
            layers=cfg.model.layers,
            pre_trained=cfg.model.pre_trained,
            n_features=params.get("n_features", cfg.model.n_features),
            image_size=image_size,
        )
        _load_checkpoint(model, find_category_checkpoint(artifact.best_artifact_path, category), device)
        spec = ModelRuntimeSpec(image_size, cfg.train.batch_size, cfg.train.num_workers, cfg.train.seed, cfg.categories or [category])
    elif model_name == "patchcore":
        cfg = load_patchcore_config(config_path)
        params = artifact.best_model_config
        image_size = int(params.get("image_size", cfg.train.image_size))
        model = build_patchcore_model(
            backbone=params.get("backbone", cfg.model.backbone),
            layers=cfg.model.layers,
            pre_trained=cfg.model.pre_trained,
            coreset_sampling_ratio=float(params.get("coreset_sampling_ratio", cfg.model.coreset_sampling_ratio)),
            num_neighbors=cfg.model.num_neighbors,
            image_size=image_size,
        )
        _load_checkpoint(model, find_category_checkpoint(artifact.best_artifact_path, category), device)
        spec = ModelRuntimeSpec(image_size, cfg.train.batch_size, cfg.train.num_workers, cfg.train.seed, cfg.categories or [category])
    elif model_name == "fastflow":
        cfg = load_fastflow_config(config_path)
        params = artifact.best_model_config
        image_size = int(params.get("image_size", cfg.train.image_size))
        model = build_fastflow_model(
            backbone=params.get("backbone", cfg.model.backbone),
            pre_trained=cfg.model.pre_trained,
            flow_steps=int(params.get("flow_steps", cfg.model.flow_steps)),
            hidden_ratio=float(params.get("hidden_ratio", cfg.model.hidden_ratio)),
            image_size=image_size,
        )
        _load_checkpoint(model, find_category_checkpoint(artifact.best_artifact_path, category), device)
        spec = ModelRuntimeSpec(image_size, cfg.train.batch_size, cfg.train.num_workers, cfg.train.seed, cfg.categories or [category])
    elif model_name == "winclip_zeroshot":
        cfg = load_winclip_config(config_path)
        model = build_winclip_zeroshot_model(class_name=category, scales=cfg.model.scales)
        image_size = int(cfg.test.image_size)
        spec = ModelRuntimeSpec(image_size, cfg.test.batch_size, cfg.test.num_workers, cfg.test.seed, cfg.categories or [category])
    else:
        raise ValueError(f"unsupported model: {model_name}")

    model.to(device)
    model.eval()
    if model_name == "winclip_zeroshot":
        # Lightning normally calls setup() before predict. Direct inference must do the same.
        model.setup("predict")
    return PredictionTupleWrapper(model.model).to(device).eval(), spec


def config_path_for_model(model_name: str) -> Path:
    root = Path(__file__).resolve().parents[3] / "config" / "anomaly_detection"
    mapping = {
        "padim": root / "padim_visa.yaml",
        "patchcore": root / "patchcore_visa.yaml",
        "fastflow": root / "fastflow_visa.yaml",
        "winclip_zeroshot": root / "winclip_zeroshot_visa.yaml",
    }
    return mapping[model_name]


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_missing = [key for key in missing if not key.startswith(("evaluator.", "pre_processor.", "post_processor."))]
    if critical_missing:
        raise RuntimeError(f"checkpoint load left critical missing keys for {ckpt_path}: {critical_missing[:20]}")
    if unexpected:
        # Keep this non-fatal because Lightning/anomalib checkpoints can include callback-facing state.
        print(f"warning: ignored unexpected checkpoint keys for {ckpt_path}: {unexpected[:10]}")


def _get_output_field(output: Any, key: str):
    if isinstance(output, dict):
        return output.get(key)
    if hasattr(output, key):
        return getattr(output, key)
    if hasattr(output, "_asdict"):
        return output._asdict().get(key)
    return None
