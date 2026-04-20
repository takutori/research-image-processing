from __future__ import annotations

import configparser
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SequenceInfo:
    name: str
    root: Path
    image_dir: Path
    gt_path: Path
    frame_rate: int
    seq_length: int
    image_width: int
    image_height: int
    image_ext: str


def resolve_dancetrack_root(dataset_root: str | Path | None = None) -> Path:
    if dataset_root is not None:
        return Path(dataset_root)
    project_root = Path(__file__).resolve().parents[3]
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    data_root = Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"
    return data_root / "dancetrack"


def read_sequence_info(sequence_root: str | Path) -> SequenceInfo:
    sequence_root = Path(sequence_root)
    parser = configparser.ConfigParser()
    parser.read(sequence_root / "seqinfo.ini")
    section = parser["Sequence"]
    return SequenceInfo(
        name=section["name"],
        root=sequence_root,
        image_dir=sequence_root / section.get("imDir", "img1"),
        gt_path=sequence_root / "gt" / "gt.txt",
        frame_rate=int(section["frameRate"]),
        seq_length=int(section["seqLength"]),
        image_width=int(section["imWidth"]),
        image_height=int(section["imHeight"]),
        image_ext=section.get("imExt", ".jpg"),
    )


def list_split_sequences(
    dataset_root: str | Path,
    split: str,
    sequence_names: list[str] | None = None,
) -> list[SequenceInfo]:
    split_root = Path(dataset_root) / split
    if not split_root.exists():
        raise FileNotFoundError(f"split root not found: {split_root}")
    names = sequence_names or sorted(path.name for path in split_root.iterdir() if path.is_dir())
    return [read_sequence_info(split_root / name) for name in names]


def select_shortest_sequences(
    sequences: list[SequenceInfo],
    count: int,
) -> list[SequenceInfo]:
    return sorted(sequences, key=lambda seq: (seq.seq_length, seq.name))[:count]


def load_mot_txt(txt_path: str | Path, with_score: bool = True) -> pd.DataFrame:
    txt_path = Path(txt_path)
    columns = ["frame", "track_id", "x", "y", "w", "h", "score", "x_world", "y_world", "z_world"]
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(txt_path, header=None)
    if frame.shape[1] < len(columns):
        for _ in range(len(columns) - frame.shape[1]):
            frame[frame.shape[1]] = -1
    frame = frame.iloc[:, : len(columns)]
    frame.columns = columns
    if not with_score:
        frame["score"] = 1.0
    return frame


def load_ground_truth(sequence: SequenceInfo, frame_limit: int | None = None) -> pd.DataFrame:
    frame = load_mot_txt(sequence.gt_path, with_score=False)
    if frame_limit is not None:
        frame = frame[frame["frame"] <= frame_limit].copy()
    frame["sequence"] = sequence.name
    return frame


def build_gt_stats(
    sequences: list[SequenceInfo],
    frame_limit: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sequence in sequences:
        gt_df = load_ground_truth(sequence, frame_limit=frame_limit)
        effective_length = min(sequence.seq_length, frame_limit) if frame_limit else sequence.seq_length
        detections_per_frame = gt_df.groupby("frame").size()
        rows.append(
            {
                "sequence": sequence.name,
                "seq_length": effective_length,
                "num_gt_boxes": int(len(gt_df)),
                "num_gt_tracks": int(gt_df["track_id"].nunique()) if not gt_df.empty else 0,
                "avg_objects_per_frame": float(detections_per_frame.mean()) if not detections_per_frame.empty else 0.0,
                "max_objects_per_frame": int(detections_per_frame.max()) if not detections_per_frame.empty else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True)


def write_trackeval_gt_root(
    sequences: list[SequenceInfo],
    destination_root: str | Path,
    frame_limit: int | None = None,
) -> Path:
    destination_root = Path(destination_root)
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        seq_root = destination_root / sequence.name
        (seq_root / "gt").mkdir(parents=True, exist_ok=True)
        gt_df = load_ground_truth(sequence, frame_limit=frame_limit)
        gt_df.loc[:, ["frame", "track_id", "x", "y", "w", "h", "score", "x_world", "y_world", "z_world"]].to_csv(
            seq_root / "gt" / "gt.txt",
            header=False,
            index=False,
        )
        parser = configparser.ConfigParser()
        parser["Sequence"] = {
            "name": sequence.name,
            "imDir": "img1",
            "frameRate": str(sequence.frame_rate),
            "seqLength": str(min(sequence.seq_length, frame_limit) if frame_limit else sequence.seq_length),
            "imWidth": str(sequence.image_width),
            "imHeight": str(sequence.image_height),
            "imExt": sequence.image_ext,
        }
        with (seq_root / "seqinfo.ini").open("w", encoding="utf-8") as file:
            parser.write(file)
    return destination_root


def get_frame_paths(sequence: SequenceInfo, frame_limit: int | None = None) -> list[Path]:
    frame_paths = sorted(sequence.image_dir.glob(f"*{sequence.image_ext}"))
    if frame_limit is not None:
        frame_paths = frame_paths[:frame_limit]
    return frame_paths


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
