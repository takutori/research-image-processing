from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_DATASET_ROOT = str(Path(os.environ.get("HOST_DATA_DIR", "")) / "visa") if os.environ.get("HOST_DATA_DIR") else str(Path(__file__).resolve().parents[3] / "data" / "visa")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "anomaly_detection" / "winclip_fewshot_visa.yaml"


@dataclass
class WinClipFewShotModelConfig:
    backbone: str = "ViT-B-16-plus-240"
    class_name: str = "object"
    scales: list[int] = field(default_factory=lambda: [2, 3])
    k_shot: int = 4


@dataclass
class WinClipFewShotTrainConfig:
    batch_size: int = 8
    num_workers: int = 0
    fewshot_split_csv: str = "2cls_fewshot.csv"


@dataclass
class WinClipFewShotTestConfig:
    batch_size: int = 8
    num_workers: int = 0
    test_split_csv: str = "1cls.csv"


@dataclass
class WinClipFewShotExperimentConfig:
    experiment_name: str = "winclip_fewshot_visa"
    dataset_root: str = DEFAULT_DATASET_ROOT
    output_dir: str = ""
    categories: list[str] | None = None
    model: WinClipFewShotModelConfig = field(default_factory=WinClipFewShotModelConfig)
    train: WinClipFewShotTrainConfig = field(default_factory=WinClipFewShotTrainConfig)
    test: WinClipFewShotTestConfig = field(default_factory=WinClipFewShotTestConfig)

    @property
    def output_path(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        base = Path(os.environ.get("HOST_DATA_DIR", "")) if os.environ.get("HOST_DATA_DIR") else Path(__file__).resolve().parents[3]
        return base / "checkpoints" / "anomaly_detection" / "winclip_fewshot" / self.experiment_name


def load_experiment_config(path: str | Path) -> WinClipFewShotExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = WinClipFewShotExperimentConfig()
    cfg.experiment_name = raw.get("experiment_name", cfg.experiment_name)
    cfg.dataset_root = raw.get("dataset_root", cfg.dataset_root)
    cfg.output_dir = raw.get("output_dir", cfg.output_dir)
    cfg.categories = raw.get("categories", cfg.categories)
    if "model" in raw:
        model_raw = raw["model"]
        cfg.model.backbone = model_raw.get("backbone", cfg.model.backbone)
        cfg.model.class_name = model_raw.get("class_name", cfg.model.class_name)
        cfg.model.scales = model_raw.get("scales", cfg.model.scales)
        cfg.model.k_shot = model_raw.get("k_shot", cfg.model.k_shot)
    if "train" in raw:
        train_raw = raw["train"]
        cfg.train.batch_size = train_raw.get("batch_size", cfg.train.batch_size)
        cfg.train.num_workers = train_raw.get("num_workers", cfg.train.num_workers)
        cfg.train.fewshot_split_csv = train_raw.get("fewshot_split_csv", cfg.train.fewshot_split_csv)
    if "test" in raw:
        test_raw = raw["test"]
        cfg.test.batch_size = test_raw.get("batch_size", cfg.test.batch_size)
        cfg.test.num_workers = test_raw.get("num_workers", cfg.test.num_workers)
        cfg.test.test_split_csv = test_raw.get("test_split_csv", cfg.test.test_split_csv)
    return cfg
