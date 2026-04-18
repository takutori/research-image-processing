from __future__ import annotations

from ultralytics import YOLO


def build_yolo11_seg_model(model_name: str) -> YOLO:
    """Load a YOLO11-seg model. Weights are auto-downloaded by ultralytics."""
    return YOLO(model_name)
