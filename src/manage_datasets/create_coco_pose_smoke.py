from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manage_datasets.setup_coco_yolo_pose import convert_coco_keypoints_to_yolo_pose
from src.pose_estimation.common.data import COCO_PERSON_CATEGORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lightweight COCO pose smoke subset with symlinks.")
    parser.add_argument("--train-images", type=int, default=64, help="Number of train images to keep.")
    parser.add_argument("--val-images", type=int, default=32, help="Number of val images to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset sampling.")
    return parser.parse_args()


def resolve_data_root() -> Path:
    project_root = PROJECT_ROOT
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    return Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sample_image_ids(annotation_data: dict, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    grouped: dict[int, list[dict]] = {}
    for annotation in annotation_data["annotations"]:
        if int(annotation.get("category_id", -1)) != 1:
            continue
        if int(annotation.get("num_keypoints", 0)) <= 0:
            continue
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)

    single_person_ids = sorted(image_id for image_id, rows in grouped.items() if len(rows) == 1)
    multi_person_ids = sorted(image_id for image_id, rows in grouped.items() if len(rows) >= 2)
    desired_multi = min(len(multi_person_ids), count // 2)
    desired_single = min(len(single_person_ids), count - desired_multi)

    selected: list[int] = []
    if desired_single:
        selected.extend(rng.sample(single_person_ids, desired_single))
    if desired_multi:
        selected.extend(rng.sample(multi_person_ids, desired_multi))

    if len(selected) < count:
        remaining_pool = sorted(set(grouped) - set(selected))
        needed = count - len(selected)
        extra = remaining_pool[:needed]
    else:
        extra = []

    selected.extend(extra)
    return sorted(selected)


def filter_coco(annotation_data: dict, image_ids: list[int], *, person_only: bool) -> dict:
    image_id_set = set(image_ids)
    filtered_images = [image for image in annotation_data["images"] if int(image["id"]) in image_id_set]
    filtered_annotations = [
        annotation
        for annotation in annotation_data["annotations"]
        if int(annotation["image_id"]) in image_id_set
        and (not person_only or int(annotation.get("category_id", -1)) == COCO_PERSON_CATEGORY["id"])
    ]
    filtered_categories = [COCO_PERSON_CATEGORY] if person_only else annotation_data["categories"]
    return {
        "info": annotation_data.get("info", {}),
        "licenses": annotation_data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories,
    }


def symlink_images(source_root: Path, smoke_root: Path, image_records: list[dict], split: str) -> None:
    target_dir = smoke_root / "images" / split
    target_dir.mkdir(parents=True, exist_ok=True)
    for image in image_records:
        source_path = source_root / "images" / split / image["file_name"]
        target_path = target_dir / image["file_name"]
        if target_path.exists() or target_path.is_symlink():
            continue
        target_path.symlink_to(source_path)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root()
    source_root = data_root / "coco"
    smoke_root = data_root / "coco_pose_smoke"

    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    (smoke_root / "annotations").mkdir(parents=True, exist_ok=True)

    train_pose = load_json(source_root / "annotations" / "person_keypoints_train2017.json")
    val_pose = load_json(source_root / "annotations" / "person_keypoints_val2017.json")
    train_det = load_json(source_root / "annotations" / "instances_train2017.json")
    val_det = load_json(source_root / "annotations" / "instances_val2017.json")

    train_image_ids = sample_image_ids(train_pose, args.train_images, args.seed)
    val_image_ids = sample_image_ids(val_pose, args.val_images, args.seed + 1)

    train_pose_subset = filter_coco(train_pose, train_image_ids, person_only=True)
    val_pose_subset = filter_coco(val_pose, val_image_ids, person_only=True)
    train_det_subset = filter_coco(train_det, train_image_ids, person_only=True)
    val_det_subset = filter_coco(val_det, val_image_ids, person_only=True)

    (smoke_root / "annotations" / "person_keypoints_train2017.json").write_text(
        json.dumps(train_pose_subset, ensure_ascii=False, indent=2)
    )
    (smoke_root / "annotations" / "person_keypoints_val2017.json").write_text(
        json.dumps(val_pose_subset, ensure_ascii=False, indent=2)
    )
    (smoke_root / "annotations" / "instances_train2017.json").write_text(
        json.dumps(train_det_subset, ensure_ascii=False, indent=2)
    )
    (smoke_root / "annotations" / "instances_val2017.json").write_text(
        json.dumps(val_det_subset, ensure_ascii=False, indent=2)
    )

    symlink_images(source_root, smoke_root, train_pose_subset["images"], "train2017")
    symlink_images(source_root, smoke_root, val_pose_subset["images"], "val2017")

    convert_coco_keypoints_to_yolo_pose(
        smoke_root / "annotations" / "person_keypoints_train2017.json",
        smoke_root / "labels" / "train2017",
    )
    convert_coco_keypoints_to_yolo_pose(
        smoke_root / "annotations" / "person_keypoints_val2017.json",
        smoke_root / "labels" / "val2017",
    )

    metadata = {
        "train_images": len(train_pose_subset["images"]),
        "val_images": len(val_pose_subset["images"]),
        "seed": args.seed,
        "dataset_root": str(smoke_root),
    }
    (smoke_root / "smoke_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print("")
    print("COCO pose smoke subset created.")
    print(f"source root: {source_root}")
    print(f"smoke root: {smoke_root}")
    print(f"train images: {len(train_pose_subset['images'])}")
    print(f"val images: {len(val_pose_subset['images'])}")


if __name__ == "__main__":
    main()
