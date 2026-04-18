from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class YOLO11TrainConfig:
    model_name: str = "yolo11s.pt"
    img_size: int = 640
    epochs: int = 50
    patience: int = 100
    batch: int = 16
    workers: int = 8
    fraction: float = 1.0
    device: str = "0"


@dataclass
class YOLO11GridConfig:
    model_names: list[str]
    img_sizes: list[int]
    epochs: int = 20
    patience: int = 100
    batch: int = 16
    workers: int = 8
    fraction: float = 1.0


@dataclass
class YOLO11ExperimentConfig:
    experiment_name: str
    dataset_yaml: str
    output_dir: str
    tuning_mode: str  # "none" | "grid"
    train: YOLO11TrainConfig
    grid: YOLO11GridConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path(
    "/workspace/config/object_detection/yolo11_coco.yaml"
)


def load_experiment_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> YOLO11ExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    train_cfg = YOLO11TrainConfig(**raw["train"])
    grid_raw = raw.get("grid", {})
    grid_cfg = YOLO11GridConfig(
        model_names=grid_raw.get("model_names", [train_cfg.model_name]),
        img_sizes=grid_raw.get("img_sizes", [train_cfg.img_size]),
        epochs=grid_raw.get("epochs", train_cfg.epochs),
        patience=grid_raw.get("patience", train_cfg.patience),
        batch=grid_raw.get("batch", train_cfg.batch),
        workers=grid_raw.get("workers", train_cfg.workers),
        fraction=grid_raw.get("fraction", train_cfg.fraction),
    )

    return YOLO11ExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_yaml=raw["dataset_yaml"],
        output_dir=raw["output_dir"],
        tuning_mode=raw.get("tuning_mode", "none"),
        train=train_cfg,
        grid=grid_cfg,
    )
