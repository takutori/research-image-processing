from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class YOLOWorldTestConfig:
    model_name: str = "yolov8s-worldv2.pt"
    img_size: int = 640
    batch_size: int = 8
    conf_threshold: float = 0.001
    iou_threshold: float = 0.6
    max_images: int | None = None
    device: str = "0"


@dataclass
class YOLOWorldExperimentConfig:
    experiment_name: str
    dataset_root: str
    output_dir: str
    test: YOLOWorldTestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path(
    "/workspace/config/object_detection/yolo_world_coco.yaml"
)


def load_experiment_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> YOLOWorldExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    test_raw = raw.get("test", {})
    test_cfg = YOLOWorldTestConfig(
        model_name=test_raw.get("model_name", "yolov8s-worldv2.pt"),
        img_size=test_raw.get("img_size", 640),
        batch_size=test_raw.get("batch_size", 8),
        conf_threshold=test_raw.get("conf_threshold", 0.001),
        iou_threshold=test_raw.get("iou_threshold", 0.6),
        max_images=test_raw.get("max_images", None),
        device=test_raw.get("device", "0"),
    )

    return YOLOWorldExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        test=test_cfg,
    )
