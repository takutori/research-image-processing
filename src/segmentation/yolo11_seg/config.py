from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class YOLO11SegTestConfig:
    model_name: str = "yolo11s-seg.pt"
    img_size: int = 640
    batch: int = 8
    device: str = "0"
    conf: float = 0.001
    iou: float = 0.6
    max_images: int | None = None


@dataclass
class YOLO11SegExperimentConfig:
    experiment_name: str
    dataset_root: str
    output_dir: str
    test: YOLO11SegTestConfig

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path(
    "/workspace/config/segmentation/yolo11_seg_coco.yaml"
)


def load_experiment_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> YOLO11SegExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    test_raw = raw.get("test", {})
    test_cfg = YOLO11SegTestConfig(
        model_name=test_raw.get("model_name", "yolo11s-seg.pt"),
        img_size=test_raw.get("img_size", 640),
        batch=test_raw.get("batch", 8),
        device=test_raw.get("device", "0"),
        conf=test_raw.get("conf", 0.001),
        iou=test_raw.get("iou", 0.6),
        max_images=test_raw.get("max_images", None),
    )

    return YOLO11SegExperimentConfig(
        experiment_name=raw["experiment_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        test=test_cfg,
    )
