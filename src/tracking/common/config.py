from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class TrackingTestConfig:
    detector_model: str = "yolo11n.pt"
    tracker_name: str = "bytetrack.yaml"
    img_size: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "0"
    sequence_names: list[str] | None = None
    auto_select_shortest_n: int | None = None
    frame_limit: int | None = None
    visualization_frame_limit: int = 600


@dataclass
class TrackingExperimentConfig:
    experiment_name: str
    dataset_root: str
    split: str
    output_dir: str
    test: TrackingTestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def load_tracking_experiment_config(
    config_path: str | Path,
) -> TrackingExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())
    test_cfg = TrackingTestConfig(**raw.get("test", {}))
    return TrackingExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        split=raw.get("split", "val"),
        output_dir=raw["output_dir"],
        test=test_cfg,
    )
