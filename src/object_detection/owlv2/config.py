from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class OWLv2TestConfig:
    model_name: str = "google/owlv2-base-patch16-ensemble"
    batch_size: int = 4
    num_workers: int = 0
    score_threshold: float = 0.1
    max_images: int | None = None
    device: str = "cuda"


@dataclass
class OWLv2ExperimentConfig:
    experiment_name: str
    dataset_root: str
    output_dir: str
    test: OWLv2TestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path(
    "/workspace/config/object_detection/owlv2_coco.yaml"
)


def load_experiment_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> OWLv2ExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    test_raw = raw.get("test", {})
    test_cfg = OWLv2TestConfig(
        model_name=test_raw.get("model_name", "google/owlv2-base-patch16-ensemble"),
        batch_size=test_raw.get("batch_size", 4),
        num_workers=test_raw.get("num_workers", 0),
        score_threshold=test_raw.get("score_threshold", 0.1),
        max_images=test_raw.get("max_images", None),
        device=test_raw.get("device", "cuda"),
    )

    return OWLv2ExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        test=test_cfg,
    )
