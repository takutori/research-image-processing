"""Setup COCO 2017 for YOLO11 training.

ultralytics resolves label paths by replacing '/images/' with '/labels/' in the
image path. The downloaded COCO images are placed directly at:

    <coco_root>/train2017/*.jpg
    <coco_root>/val2017/*.jpg

which does not contain '/images/', so labels would be looked for in the same
directory as the images (wrong).

Using directory symlinks for images is not robust because some runtime paths
are resolved to the original target directory (without '/images/'), which makes
ultralytics look for labels in the wrong place.

This script fixes that by:
  1. Creating image directories under <coco_root>/images/ and populating them
     with hardlinks to the original COCO image files
  2. Converting COCO JSON annotations to YOLO-format .txt files under
                        <coco_root>/labels/train2017/
                        <coco_root>/labels/val2017/
  3. Updating config/object_detection/coco_dataset.yaml to use
     train: images/train2017  /  val: images/val2017
  4. Removing ultralytics dataset cache files so the new directory layout is
     re-scanned on the next training run

Prerequisites:
  - train2017/ and val2017/ image directories must exist
  - annotations/instances_train2017.json and annotations/instances_val2017.json
    must exist  (run: uv run python src/manage_datasets/download_coco.py --parts annotations)

Usage (from /workspace):
  uv run python src/manage_datasets/setup_coco_yolo.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def resolve_coco_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    base = Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"
    return base / "coco"


def create_images_layout(coco_root: Path) -> None:
    """Create images/<split>/ directories populated with hardlinked images."""
    images_dir = coco_root / "images"
    images_dir.mkdir(exist_ok=True)

    for split in ("train2017", "val2017"):
        split_dir = images_dir / split
        source_dir = coco_root / split

        if not source_dir.exists():
            print(f"skip images layout: {split} source not found at {source_dir}")
            continue

        if split_dir.is_symlink():
            split_dir.unlink()
            print(f"removed symlink: {split_dir}")
        elif split_dir.exists() and not split_dir.is_dir():
            raise RuntimeError(f"expected directory path, found file: {split_dir}")

        split_dir.mkdir(parents=True, exist_ok=True)

        linked = 0
        skipped = 0
        for image_path in sorted(source_dir.glob("*.jpg")):
            target_path = split_dir / image_path.name
            if target_path.exists():
                skipped += 1
                continue

            try:
                os.link(image_path, target_path)
            except OSError:
                shutil.copy2(image_path, target_path)
            linked += 1

        print(
            f"prepared images/{split}: created {linked} files, skipped {skipped} existing files"
        )


def convert_coco_to_yolo(ann_file: Path, labels_dir: Path) -> None:
    """Convert COCO JSON annotations to YOLO-format .txt files.

    YOLO format per line: <class_id> <cx> <cy> <w> <h>  (all normalised 0-1)
    COCO bbox format: [x_min, y_min, width, height] in pixels.
    """
    if not ann_file.exists():
        print(f"skip convert: {ann_file} not found")
        return

    # Check if already converted (heuristic: labels_dir has at least 1 .txt)
    if labels_dir.exists() and any(labels_dir.glob("*.txt")):
        print(f"skip convert: {labels_dir} already has label files")
        return

    print(f"converting: {ann_file.name} -> {labels_dir}")
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        data = json.load(f)

    # Build category_id -> 0-based class index mapping (COCO 80-class order)
    cat_id_to_cls: dict[int, int] = {
        cat["id"]: idx for idx, cat in enumerate(
            sorted(data["categories"], key=lambda c: c["id"])
        )
    }

    # Build image_id -> (width, height, file_stem) mapping
    img_info: dict[int, tuple[int, int, str]] = {
        img["id"]: (img["width"], img["height"], Path(img["file_name"]).stem)
        for img in data["images"]
    }

    # Collect annotations per image
    per_image: dict[int, list[str]] = {img_id: [] for img_id in img_info}
    skipped = 0

    for ann in data["annotations"]:
        # Skip crowd annotations
        if ann.get("iscrowd", 0):
            skipped += 1
            continue

        image_id = ann["image_id"]
        cat_id = ann["category_id"]
        if image_id not in img_info or cat_id not in cat_id_to_cls:
            skipped += 1
            continue

        img_w, img_h, _ = img_info[image_id]
        cls_idx = cat_id_to_cls[cat_id]
        x_min, y_min, bw, bh = ann["bbox"]

        # Convert to normalised cx, cy, w, h
        cx = (x_min + bw / 2) / img_w
        cy = (y_min + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        per_image[image_id].append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # Write one .txt per image (even empty = background-only image)
    written = 0
    for image_id, lines in per_image.items():
        _, _, stem = img_info[image_id]
        label_file = labels_dir / f"{stem}.txt"
        label_file.write_text("\n".join(lines))
        written += 1

    print(f"  wrote {written} label files ({skipped} annotations skipped)")


def update_coco_dataset_yaml(project_root: Path) -> None:
    """Rewrite coco_dataset.yaml to use images/train2017 and images/val2017."""
    yaml_path = project_root / "config" / "object_detection" / "coco_dataset.yaml"
    if not yaml_path.exists():
        print(f"skip yaml update: {yaml_path} not found")
        return

    text = yaml_path.read_text()

    updated = text
    updated = updated.replace("train: train2017", "train: images/train2017")
    updated = updated.replace("val: val2017", "val: images/val2017")

    if updated == text:
        print("skip yaml update: coco_dataset.yaml already up-to-date")
        return

    yaml_path.write_text(updated)
    print(f"updated: {yaml_path}")


def remove_ultralytics_caches(coco_root: Path) -> None:
    """Remove dataset cache files so ultralytics re-scans the fixed layout."""
    for cache_path in (coco_root / "train2017.cache", coco_root / "val2017.cache"):
        if cache_path.exists():
            cache_path.unlink()
            print(f"removed cache: {cache_path}")
        else:
            print(f"skip cache removal: {cache_path} not found")


def main() -> None:
    coco_root = resolve_coco_root()
    project_root = Path(__file__).resolve().parents[2]

    print(f"coco_root: {coco_root}")

    if not coco_root.exists():
        print("ERROR: coco_root does not exist. Run download_coco.py first.")
        sys.exit(1)

    # Step 1: images layout
    create_images_layout(coco_root)

    # Step 2: YOLO label conversion
    for split in ("train2017", "val2017"):
        ann_file = coco_root / "annotations" / f"instances_{split}.json"
        labels_dir = coco_root / "labels" / split
        convert_coco_to_yolo(ann_file, labels_dir)

    # Step 3: update coco_dataset.yaml
    update_coco_dataset_yaml(project_root)

    # Step 4: remove caches
    remove_ultralytics_caches(coco_root)

    print("\nsetup complete.")
    print("  images: <coco_root>/images/train2017/")
    print("  images: <coco_root>/images/val2017/")
    print("  labels: <coco_root>/labels/train2017/")
    print("  labels: <coco_root>/labels/val2017/")


if __name__ == "__main__":
    main()
