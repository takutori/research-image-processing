from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class SAM2TestConfig:
    checkpoint: str = "/workspace/checkpoints/segmentation/sam2/sam2.1_hiera_large.pt"
    model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device: str = "cuda"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.92
    max_images: int | None = None


@dataclass
class SAM2ExperimentConfig:
    experiment_name: str
    dataset_root: str
    output_dir: str
    test: SAM2TestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path("/workspace/config/segmentation/sam2_coco.yaml")


def load_experiment_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> SAM2ExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    test_raw = raw.get("test", {})
    test_cfg = SAM2TestConfig(
        checkpoint=test_raw.get(
            "checkpoint",
            "/workspace/checkpoints/segmentation/sam2/sam2.1_hiera_large.pt",
        ),
        model_config=test_raw.get("model_config", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        device=test_raw.get("device", "cuda"),
        points_per_side=test_raw.get("points_per_side", 32),
        pred_iou_thresh=test_raw.get("pred_iou_thresh", 0.7),
        stability_score_thresh=test_raw.get("stability_score_thresh", 0.92),
        max_images=test_raw.get("max_images", None),
    )

    return SAM2ExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        test=test_cfg,
    )
