from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_DATASET_ROOT = str(Path(os.environ.get("HOST_DATA_DIR", "")) / "visa") if os.environ.get("HOST_DATA_DIR") else str(Path(__file__).resolve().parents[3] / "data" / "visa")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "anomaly_detection" / "patchcore_visa.yaml"


@dataclass
class PatchCoreModelConfig:
    backbone: str = "wide_resnet50_2"
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    pre_trained: bool = True
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9


@dataclass
class PatchCoreTrainConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 12
    val_split_ratio: float = 0.2
    seed: int = 42


@dataclass
class PatchCoreGridConfig:
    backbones: list[str] = field(default_factory=lambda: ["wide_resnet50_2", "resnet18"])
    image_sizes: list[int] = field(default_factory=lambda: [224, 320])
    coreset_sampling_ratios: list[float] = field(default_factory=lambda: [0.1])


@dataclass
class PatchCoreExperimentConfig:
    experiment_name: str = "patchcore_visa"
    dataset_root: str = DEFAULT_DATASET_ROOT
    output_dir: str = ""
    tuning_mode: str = "grid"
    categories: list[str] | None = None
    model: PatchCoreModelConfig = field(default_factory=PatchCoreModelConfig)
    train: PatchCoreTrainConfig = field(default_factory=PatchCoreTrainConfig)
    grid: PatchCoreGridConfig = field(default_factory=PatchCoreGridConfig)

    @property
    def output_path(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        base = Path(os.environ.get("HOST_DATA_DIR", "")) if os.environ.get("HOST_DATA_DIR") else Path(__file__).resolve().parents[3]
        return base / "checkpoints" / "anomaly_detection" / "patchcore" / self.experiment_name


def load_experiment_config(path: str | Path) -> PatchCoreExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = PatchCoreExperimentConfig()
    cfg.experiment_name = raw.get("experiment_name", cfg.experiment_name)
    cfg.dataset_root = raw.get("dataset_root", cfg.dataset_root)
    cfg.output_dir = raw.get("output_dir", cfg.output_dir)
    cfg.tuning_mode = raw.get("tuning_mode", cfg.tuning_mode)
    cfg.categories = raw.get("categories", cfg.categories)
    if "model" in raw:
        m = raw["model"]
        cfg.model.backbone = m.get("backbone", cfg.model.backbone)
        cfg.model.layers = m.get("layers", cfg.model.layers)
        cfg.model.pre_trained = m.get("pre_trained", cfg.model.pre_trained)
        cfg.model.coreset_sampling_ratio = m.get("coreset_sampling_ratio", cfg.model.coreset_sampling_ratio)
        cfg.model.num_neighbors = m.get("num_neighbors", cfg.model.num_neighbors)
    if "train" in raw:
        t = raw["train"]
        cfg.train.image_size = t.get("image_size", cfg.train.image_size)
        cfg.train.batch_size = t.get("batch_size", cfg.train.batch_size)
        cfg.train.num_workers = t.get("num_workers", cfg.train.num_workers)
        cfg.train.val_split_ratio = t.get("val_split_ratio", cfg.train.val_split_ratio)
        cfg.train.seed = t.get("seed", cfg.train.seed)
    if "grid" in raw:
        g = raw["grid"]
        cfg.grid.backbones = g.get("backbones", cfg.grid.backbones)
        cfg.grid.image_sizes = g.get("image_sizes", cfg.grid.image_sizes)
        cfg.grid.coreset_sampling_ratios = g.get("coreset_sampling_ratios", cfg.grid.coreset_sampling_ratios)
    return cfg
