from __future__ import annotations

from ultralytics import YOLO


def build_yolo11_model(model_name: str = "yolo11s.pt") -> YOLO:
    """Load a YOLO11 model by name or path.

    Args:
        model_name: Pretrained model name (e.g. "yolo11s.pt") or path to .pt.

    Returns:
        ultralytics YOLO instance.
    """
    return YOLO(model_name)
