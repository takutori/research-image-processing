from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class GroundedSAMTestConfig:
    detector_model: str = "IDEA-Research/grounding-dino-base"
    sam2_checkpoint: str = "/workspace/checkpoints/segmentation/sam2/sam2.1_hiera_large.pt"
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device: str = "cuda"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    chunk_size: int = 20
    max_images: int | None = None


@dataclass
class GroundedSAMExperimentConfig:
    experiment_name: str
    dataset_root: str
    output_dir: str
    test: GroundedSAMTestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path("/workspace/config/segmentation/grounded_sam_coco.yaml")


def load_experiment_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> GroundedSAMExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    test_raw = raw.get("test", {})
    test_cfg = GroundedSAMTestConfig(
        detector_model=test_raw.get("detector_model", "IDEA-Research/grounding-dino-base"),
        sam2_checkpoint=test_raw.get(
            "sam2_checkpoint",
            "/workspace/checkpoints/segmentation/sam2/sam2.1_hiera_large.pt",
        ),
        sam2_config=test_raw.get("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        device=test_raw.get("device", "cuda"),
        box_threshold=test_raw.get("box_threshold", 0.3),
        text_threshold=test_raw.get("text_threshold", 0.25),
        chunk_size=test_raw.get("chunk_size", 20),
        max_images=test_raw.get("max_images", None),
    )

    return GroundedSAMExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        test=test_cfg,
    )
