from __future__ import annotations

from ultralytics import YOLOWorld


def build_yolo_world_model(model_name: str = "yolov8s-worldv2.pt") -> YOLOWorld:
    """Load a YOLO-World model.

    Args:
        model_name: Model name or path (e.g. "yolov8s-worldv2.pt").

    Returns:
        ultralytics YOLOWorld instance.
    """
    return YOLOWorld(model_name)
