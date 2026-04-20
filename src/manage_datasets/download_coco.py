from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


COCO_DOWNLOADS = {
    "train": {
        "filename": "train2017.zip",
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "expected_paths": ["train2017"],
    },
    "val": {
        "filename": "val2017.zip",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "expected_paths": ["val2017"],
    },
    "test": {
        "filename": "test2017.zip",
        "url": "http://images.cocodataset.org/zips/test2017.zip",
        "expected_paths": ["test2017"],
    },
    "annotations": {
        "filename": "annotations_trainval2017.zip",
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "expected_paths": ["annotations"],
    },
    "image_info_test": {
        "filename": "image_info_test2017.zip",
        "url": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
        "expected_paths": ["annotations/image_info_test2017.json"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download COCO 2017 dataset files.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["train", "val", "test", "annotations", "image_info_test", "all"],
        default=["all"],
        help="Download target parts. Default: all",
    )
    return parser.parse_args()


def resolve_data_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    return Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"skip download: {destination.name}")
        return

    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded / total_size * 100
        sys.stdout.write(
            f"\rdownloading {destination.name}: {percent:6.2f}% "
            f"({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)"
        )
        sys.stdout.flush()

    print(f"start download: {url}")
    urllib.request.urlretrieve(url, destination, reporthook=reporthook)
    sys.stdout.write("\n")


def is_already_extracted(destination_dir: Path, expected_paths: list[str]) -> bool:
    return all((destination_dir / expected_path).exists() for expected_path in expected_paths)


def extract_zip(archive_path: Path, destination_dir: Path, expected_paths: list[str]) -> None:
    if is_already_extracted(destination_dir, expected_paths):
        print(f"skip extract: {archive_path.name}")
        return

    print(f"extracting: {archive_path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination_dir)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root()
    coco_root = data_root / "coco"
    downloads_root = coco_root / "_downloads"

    coco_root.mkdir(parents=True, exist_ok=True)
    downloads_root.mkdir(parents=True, exist_ok=True)

    selected_parts = ["train", "val", "test", "annotations", "image_info_test"]
    if "all" not in args.parts:
        selected_parts = args.parts

    for part in selected_parts:
        item = COCO_DOWNLOADS[part]
        archive_path = downloads_root / item["filename"]
        download_file(item["url"], archive_path)
        extract_zip(archive_path, coco_root, item["expected_paths"])

    print("")
    print("COCO 2017 download completed.")
    print(f"dataset root: {coco_root}")
    print("downloaded parts:")
    for part in selected_parts:
        print(f" - {part}")


if __name__ == "__main__":
    main()
