from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class GroundingDINOTestConfig:
    model_name: str = "IDEA-Research/grounding-dino-tiny"
    batch_size: int = 4
    num_workers: int = 0
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    chunk_size: int = 20  # number of classes per text prompt chunk
    max_images: int | None = None
    device: str = "cuda"


@dataclass
class GroundingDINOExperimentConfig:
    experiment_name: str
    dataset_root: str
    output_dir: str
    test: GroundingDINOTestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path(
    "/workspace/config/object_detection/grounding_dino_coco.yaml"
)


def load_experiment_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> GroundingDINOExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    test_raw = raw.get("test", {})
    test_cfg = GroundingDINOTestConfig(
        model_name=test_raw.get("model_name", "IDEA-Research/grounding-dino-tiny"),
        batch_size=test_raw.get("batch_size", 4),
        num_workers=test_raw.get("num_workers", 0),
        box_threshold=test_raw.get("box_threshold", 0.3),
        text_threshold=test_raw.get("text_threshold", 0.25),
        chunk_size=test_raw.get("chunk_size", 20),
        max_images=test_raw.get("max_images", None),
        device=test_raw.get("device", "cuda"),
    )

    return GroundingDINOExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        test=test_cfg,
    )
