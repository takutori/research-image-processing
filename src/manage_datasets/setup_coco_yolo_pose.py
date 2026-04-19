"""Setup COCO person keypoints as a dedicated Ultralytics pose dataset.

This creates a separate dataset root so pose labels do not conflict with
existing detection labels under ``data/coco``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.common.data import COCO_PERSON_FLIP_IDX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup COCO person keypoints for YOLO11-pose.")
    parser.add_argument("--source-dataset", default="coco", help="Source dataset root name under HOST_DATA_DIR/data.")
    parser.add_argument("--target-dataset", default="coco_pose", help="Target dataset root name under HOST_DATA_DIR/data.")
    return parser.parse_args()


def resolve_data_root() -> Path:
    project_root = PROJECT_ROOT
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    return Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"


def create_images_layout(source_root: Path, target_root: Path) -> None:
    for split in ("train2017", "val2017"):
        source_images_dir = source_root / "images" / split
        target_images_dir = target_root / "images" / split
        target_images_dir.mkdir(parents=True, exist_ok=True)

        linked = 0
        skipped = 0
        for image_path in sorted(source_images_dir.glob("*.jpg")):
            target_path = target_images_dir / image_path.name
            if target_path.exists():
                skipped += 1
                continue
            try:
                os.link(image_path, target_path)
            except OSError:
                shutil.copy2(image_path, target_path)
            linked += 1
        print(f"prepared images/{split}: created {linked} files, skipped {skipped} existing files")


def convert_coco_keypoints_to_yolo_pose(ann_file: Path, labels_dir: Path) -> None:
    if not ann_file.exists():
        raise FileNotFoundError(f"annotation file not found: {ann_file}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(ann_file.read_text())
    image_lookup = {
        int(image["id"]): (int(image["width"]), int(image["height"]), Path(image["file_name"]).stem)
        for image in data["images"]
    }
    per_image: dict[int, list[str]] = {image_id: [] for image_id in image_lookup}

    skipped = 0
    for annotation in data["annotations"]:
        if int(annotation.get("category_id", -1)) != 1 or int(annotation.get("iscrowd", 0)) != 0:
            skipped += 1
            continue
        image_id = int(annotation["image_id"])
        width, height, _ = image_lookup[image_id]
        x, y, w, h = [float(value) for value in annotation["bbox"]]
        cx = max(0.0, min(1.0, (x + w / 2.0) / width))
        cy = max(0.0, min(1.0, (y + h / 2.0) / height))
        nw = max(0.0, min(1.0, w / width))
        nh = max(0.0, min(1.0, h / height))

        keypoints = annotation.get("keypoints", [])
        normalized: list[str] = []
        for index in range(0, len(keypoints), 3):
            kx = float(keypoints[index])
            ky = float(keypoints[index + 1])
            visible = int(keypoints[index + 2])
            if visible > 0:
                nkx = max(0.0, min(1.0, kx / width))
                nky = max(0.0, min(1.0, ky / height))
            else:
                nkx = 0.0
                nky = 0.0
            normalized.extend([f"{nkx:.6f}", f"{nky:.6f}", str(visible)])

        per_image[image_id].append(
            " ".join(["0", f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}", *normalized])
        )

    written = 0
    for image_id, lines in per_image.items():
        _, _, stem = image_lookup[image_id]
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
        written += 1
    print(f"converted {ann_file.name}: wrote {written} label files ({skipped} annotations skipped)")


def link_annotations(source_root: Path, target_root: Path) -> None:
    target_annotations = target_root / "annotations"
    target_annotations.mkdir(parents=True, exist_ok=True)
    for filename in (
        "person_keypoints_train2017.json",
        "person_keypoints_val2017.json",
        "instances_train2017.json",
        "instances_val2017.json",
    ):
        source_path = source_root / "annotations" / filename
        target_path = target_annotations / filename
        if target_path.exists() or target_path.is_symlink():
            continue
        target_path.symlink_to(source_path)


def write_dataset_yaml(project_root: Path, dataset_name: str, yaml_name: str) -> None:
    yaml_path = project_root / "config" / "pose_estimation" / yaml_name
    yaml_text = "\n".join(
        [
            f"path: /workspace/data/{dataset_name}",
            "train: images/train2017",
            "val: images/val2017",
            "kpt_shape: [17, 3]",
            f"flip_idx: {COCO_PERSON_FLIP_IDX}",
            "names:",
            "  0: person",
            "",
        ]
    )
    yaml_path.write_text(yaml_text)
    print(f"updated: {yaml_path}")


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root()
    project_root = Path(__file__).resolve().parents[2]
    source_root = data_root / args.source_dataset
    target_root = data_root / args.target_dataset

    target_root.mkdir(parents=True, exist_ok=True)
    create_images_layout(source_root, target_root)
    link_annotations(source_root, target_root)
    for split in ("train2017", "val2017"):
        convert_coco_keypoints_to_yolo_pose(
            source_root / "annotations" / f"person_keypoints_{split}.json",
            target_root / "labels" / split,
        )

    write_dataset_yaml(project_root, args.target_dataset, "coco_pose_dataset.yaml")
    print("")
    print("COCO YOLO pose setup completed.")
    print(f"source root: {source_root}")
    print(f"target root: {target_root}")


if __name__ == "__main__":
    main()
