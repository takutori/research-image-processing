from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class RTMPoseTrainConfig:
    config_path: str = "body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py"
    checkpoint: str | None = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth"
    learning_rate: float = 4e-3
    max_epochs: int = 20
    val_interval: int = 5
    device: str = "cuda:0"


@dataclass
class RTMPoseGridConfig:
    config_paths: list[str]
    learning_rates: list[float]
    max_epochs: int = 20


@dataclass
class RTMPoseOptunaConfig:
    n_trials: int = 10
    learning_rate_low: float = 1e-4
    learning_rate_high: float = 5e-3
    max_epochs: int = 20


@dataclass
class RTMPoseTestConfig:
    bbox_file: str | None = None
    max_images: int | None = None
    device: str = "cuda:0"


@dataclass
class RTMPoseExperimentConfig:
    experiment_name: str
    dataset_root: str
    output_dir: str
    tuning_mode: str
    train: RTMPoseTrainConfig
    grid: RTMPoseGridConfig
    optuna: RTMPoseOptunaConfig
    test: RTMPoseTestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path("/workspace/config/pose_estimation/rtmpose_coco.yaml")


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> RTMPoseExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    train_cfg = RTMPoseTrainConfig(**raw["train"])
    grid_cfg = RTMPoseGridConfig(**raw["grid"])
    optuna_cfg = RTMPoseOptunaConfig(**raw["optuna"])
    test_cfg = RTMPoseTestConfig(**raw.get("test", {}))
    return RTMPoseExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        tuning_mode=raw.get("tuning_mode", "none"),
        train=train_cfg,
        grid=grid_cfg,
        optuna=optuna_cfg,
        test=test_cfg,
    )
