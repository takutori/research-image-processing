from __future__ import annotations

from ultralytics import YOLO


def build_yolo11_pose_model(model_name: str = "yolo11n-pose.pt") -> YOLO:
    return YOLO(model_name)
