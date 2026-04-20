from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_DATASET_ROOT = str(Path(os.environ.get("HOST_DATA_DIR", "")) / "visa") if os.environ.get("HOST_DATA_DIR") else str(Path(__file__).resolve().parents[3] / "data" / "visa")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "anomaly_detection" / "winclip_zeroshot_visa.yaml"


@dataclass
class WinClipZeroShotModelConfig:
    scales: list[int] = field(default_factory=lambda: [2, 3])


@dataclass
class WinClipZeroShotTestConfig:
    image_size: int = 240
    batch_size: int = 32
    num_workers: int = 12
    seed: int = 42


@dataclass
class WinClipZeroShotExperimentConfig:
    experiment_name: str = "winclip_zeroshot_visa"
    dataset_root: str = DEFAULT_DATASET_ROOT
    output_dir: str = ""
    categories: list[str] | None = None
    model: WinClipZeroShotModelConfig = field(default_factory=WinClipZeroShotModelConfig)
    test: WinClipZeroShotTestConfig = field(default_factory=WinClipZeroShotTestConfig)

    @property
    def output_path(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        base = Path(os.environ.get("HOST_DATA_DIR", "")) if os.environ.get("HOST_DATA_DIR") else Path(__file__).resolve().parents[3]
        return base / "checkpoints" / "anomaly_detection" / "winclip_zeroshot" / self.experiment_name


def load_experiment_config(path: str | Path) -> WinClipZeroShotExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = WinClipZeroShotExperimentConfig()
    cfg.experiment_name = raw.get("experiment_name", cfg.experiment_name)
    cfg.dataset_root = raw.get("dataset_root", cfg.dataset_root)
    cfg.output_dir = raw.get("output_dir", cfg.output_dir)
    cfg.categories = raw.get("categories", cfg.categories)
    if "model" in raw:
        m = raw["model"]
        cfg.model.scales = m.get("scales", cfg.model.scales)
    test_raw = raw.get("test", raw.get("eval"))
    if test_raw:
        cfg.test.image_size = test_raw.get("image_size", cfg.test.image_size)
        cfg.test.batch_size = test_raw.get("batch_size", cfg.test.batch_size)
        cfg.test.num_workers = test_raw.get("num_workers", cfg.test.num_workers)
        cfg.test.seed = test_raw.get("seed", cfg.test.seed)
    return cfg
