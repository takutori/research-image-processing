from __future__ import annotations

from pathlib import Path

from src.tracking.common.data import SequenceInfo, get_frame_paths


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ModuleNotFoundError(
            "Tracking requires ultralytics at runtime. Run `uv sync` before executing tracking scripts."
        ) from error
    return YOLO


def run_yolo_tracking_on_sequence(
    sequence: SequenceInfo,
    detector_model: str,
    tracker_name: str,
    img_size: int,
    conf: float,
    iou: float,
    device: str,
    frame_limit: int | None = None,
) -> list[dict]:
    YOLO = require_ultralytics()
    model = YOLO(detector_model)
    frame_paths = [str(path) for path in get_frame_paths(sequence, frame_limit=frame_limit)]
    if not frame_paths:
        return []

    rows: list[dict] = []
    results = model.track(
        source=frame_paths,
        tracker=tracker_name,
        stream=True,
        persist=True,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        device=device,
        classes=[0],
        verbose=False,
        save=False,
    )

    for frame_path, result in zip(frame_paths, results):
        frame_number = int(Path(frame_path).stem)
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        for track_id, score, coords in zip(track_ids, scores, xyxy):
            x1, y1, x2, y2 = coords.tolist()
            rows.append(
                {
                    "sequence": sequence.name,
                    "frame": frame_number,
                    "track_id": int(track_id),
                    "x": float(x1),
                    "y": float(y1),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "score": float(score),
                }
            )
    return rows
