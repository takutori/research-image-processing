from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.tracking.common.data import SequenceInfo, get_frame_paths, load_ground_truth, load_mot_txt


def require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise ModuleNotFoundError(
            "Tracking visualization requires OpenCV at runtime. Run `uv sync` before executing tracking scripts."
        ) from error
    return cv2


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    base = abs(int(track_id))
    return (
        (37 * base + 17) % 255,
        (97 * base + 91) % 255,
        (53 * base + 177) % 255,
    )


def render_sequence_video(
    sequence: SequenceInfo,
    track_df: pd.DataFrame,
    output_path: str | Path,
    frame_limit: int | None = None,
    title: str | None = None,
) -> Path:
    cv2 = require_cv2()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = get_frame_paths(sequence, frame_limit=frame_limit)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        sequence.frame_rate,
        (sequence.image_width, sequence.image_height),
    )

    track_by_frame = {
        int(frame_number): frame_df.to_dict("records")
        for frame_number, frame_df in track_df.groupby("frame")
    }
    for frame_path in frame_paths:
        frame_number = int(frame_path.stem)
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        if title:
            cv2.putText(image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        for row in track_by_frame.get(frame_number, []):
            x1 = int(round(float(row["x"])))
            y1 = int(round(float(row["y"])))
            x2 = int(round(float(row["x"]) + float(row["w"])))
            y2 = int(round(float(row["y"]) + float(row["h"])))
            color = _color_for_track(int(row["track_id"]))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"id={int(row['track_id'])}"
            if "score" in row and pd.notna(row["score"]):
                label += f" {float(row['score']):.2f}"
            cv2.putText(image, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(image)
    writer.release()
    return output_path


def generate_experiment_videos(
    sequences: list[SequenceInfo],
    selected_sequences: dict[str, str],
    mot_txt_root: str | Path,
    output_dir: str | Path,
    frame_limit: int | None,
    tracker_video_name: str,
) -> None:
    output_dir = Path(output_dir) / "best_test_results" / "videos"
    mot_txt_root = Path(mot_txt_root)
    sequence_by_name = {sequence.name: sequence for sequence in sequences}

    for tag, sequence_name in selected_sequences.items():
        sequence = sequence_by_name[sequence_name]
        gt_df = load_ground_truth(sequence, frame_limit=frame_limit)
        pred_df = load_mot_txt(mot_txt_root / f"{sequence_name}.txt")
        if frame_limit is not None:
            pred_df = pred_df[pred_df["frame"] <= frame_limit].copy()
        tag_root = output_dir / tag
        render_sequence_video(sequence, gt_df, tag_root / "gt.mp4", frame_limit=frame_limit, title=f"GT {sequence_name}")
        render_sequence_video(
            sequence,
            pred_df,
            tag_root / f"{tracker_video_name}.mp4",
            frame_limit=frame_limit,
            title=f"{tracker_video_name} {sequence_name}",
        )


def ensure_sequence_video(
    sequence: SequenceInfo,
    gt_output_path: str | Path,
    pred_output_path: str | Path,
    prediction_txt_path: str | Path,
    frame_limit: int | None,
    tracker_video_name: str,
) -> None:
    gt_output_path = Path(gt_output_path)
    pred_output_path = Path(pred_output_path)
    if not gt_output_path.exists():
        gt_df = load_ground_truth(sequence, frame_limit=frame_limit)
        render_sequence_video(sequence, gt_df, gt_output_path, frame_limit=frame_limit, title=f"GT {sequence.name}")
    if not pred_output_path.exists():
        pred_df = load_mot_txt(prediction_txt_path)
        if frame_limit is not None:
            pred_df = pred_df[pred_df["frame"] <= frame_limit].copy()
        render_sequence_video(
            sequence,
            pred_df,
            pred_output_path,
            frame_limit=frame_limit,
            title=f"{tracker_video_name} {sequence.name}",
        )
