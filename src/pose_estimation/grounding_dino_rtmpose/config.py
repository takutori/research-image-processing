from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class GroundingDINORTMPoseTestConfig:
    detector_model: str = "IDEA-Research/grounding-dino-tiny"
    rtmpose_config_path: str = "body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py"
    rtmpose_checkpoint: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    crop_margin: float = 0.0
    max_images: int | None = None
    device: str = "cuda:0"


@dataclass
class GroundingDINORTMPoseExperimentConfig:
    experiment_name: str
    dataset_root: str
    output_dir: str
    test: GroundingDINORTMPoseTestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path("/workspace/config/pose_estimation/grounding_dino_rtmpose_coco.yaml")


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> GroundingDINORTMPoseExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())
    test_cfg = GroundingDINORTMPoseTestConfig(**raw.get("test", {}))
    return GroundingDINORTMPoseExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        test=test_cfg,
    )
