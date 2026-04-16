from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_DATASET_ROOT = str(Path(os.environ.get("HOST_DATA_DIR", "")) / "visa") if os.environ.get("HOST_DATA_DIR") else str(Path(__file__).resolve().parents[3] / "data" / "visa")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "anomaly_detection" / "stfpm_visa.yaml"


@dataclass
class StfpmModelConfig:
    backbone: str = "resnet18"
    layers: list[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])
    pre_trained: bool = True


@dataclass
class StfpmTrainConfig:
    image_size: int = 256
    batch_size: int = 32
    num_workers: int = 12
    max_epochs: int = 100
    val_split_ratio: float = 0.2
    seed: int = 42


@dataclass
class StfpmGridConfig:
    backbones: list[str] = field(default_factory=lambda: ["resnet18", "wide_resnet50_2"])
    image_sizes: list[int] = field(default_factory=lambda: [256])
    layers_list: list[list[str]] = field(default_factory=lambda: [["layer1", "layer2", "layer3"]])


@dataclass
class StfpmOptunaConfig:
    n_trials: int = 20
    backbone_candidates: list[str] | None = None
    image_size_candidates: list[int] | None = None
    layers_candidates: list[list[str]] | None = None


@dataclass
class StfpmExperimentConfig:
    experiment_name: str = "stfpm_visa"
    dataset_root: str = DEFAULT_DATASET_ROOT
    output_dir: str = ""
    tuning_mode: str = "full"
    categories: list[str] | None = None
    model: StfpmModelConfig = field(default_factory=StfpmModelConfig)
    train: StfpmTrainConfig = field(default_factory=StfpmTrainConfig)
    grid: StfpmGridConfig = field(default_factory=StfpmGridConfig)
    optuna: StfpmOptunaConfig = field(default_factory=StfpmOptunaConfig)

    @property
    def output_path(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        base = Path(os.environ.get("HOST_DATA_DIR", "")) if os.environ.get("HOST_DATA_DIR") else Path(__file__).resolve().parents[3]
        return base / "checkpoints" / "anomaly_detection" / "stfpm" / self.experiment_name


def load_experiment_config(path: str | Path) -> StfpmExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = StfpmExperimentConfig()
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
    if "train" in raw:
        t = raw["train"]
        cfg.train.image_size = t.get("image_size", cfg.train.image_size)
        cfg.train.batch_size = t.get("batch_size", cfg.train.batch_size)
        cfg.train.num_workers = t.get("num_workers", cfg.train.num_workers)
        cfg.train.max_epochs = t.get("max_epochs", cfg.train.max_epochs)
        cfg.train.val_split_ratio = t.get("val_split_ratio", cfg.train.val_split_ratio)
        cfg.train.seed = t.get("seed", cfg.train.seed)
    if "grid" in raw:
        g = raw["grid"]
        cfg.grid.backbones = g.get("backbones", cfg.grid.backbones)
        cfg.grid.image_sizes = g.get("image_sizes", cfg.grid.image_sizes)
        cfg.grid.layers_list = g.get("layers_list", cfg.grid.layers_list)
    if "optuna" in raw:
        o = raw["optuna"]
        cfg.optuna.n_trials = o.get("n_trials", cfg.optuna.n_trials)
        cfg.optuna.backbone_candidates = o.get("backbone_candidates", cfg.optuna.backbone_candidates)
        cfg.optuna.image_size_candidates = o.get("image_size_candidates", cfg.optuna.image_size_candidates)
        cfg.optuna.layers_candidates = o.get("layers_candidates", cfg.optuna.layers_candidates)
    return cfg
