from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


PET_DOWNLOADS = {
    "images": {
        "filename": "images.tar.gz",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        "expected_paths": ["images"],
    },
    "annotations": {
        "filename": "annotations.tar.gz",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
        "expected_paths": ["annotations", "annotations/trainval.txt", "annotations/test.txt"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Oxford-IIIT Pet dataset files.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["images", "annotations", "all"],
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


def extract_archive(archive_path: Path, destination_dir: Path, expected_paths: list[str]) -> None:
    if is_already_extracted(destination_dir, expected_paths):
        print(f"skip extract: {archive_path.name}")
        return

    print(f"extracting: {archive_path.name}")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(destination_dir)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root()
    pet_root = data_root / "oxford_iiit_pet"
    downloads_root = pet_root / "_downloads"

    pet_root.mkdir(parents=True, exist_ok=True)
    downloads_root.mkdir(parents=True, exist_ok=True)

    selected_parts = ["images", "annotations"]
    if "all" not in args.parts:
        selected_parts = args.parts

    for part in selected_parts:
        item = PET_DOWNLOADS[part]
        archive_path = downloads_root / item["filename"]
        download_file(item["url"], archive_path)
        extract_archive(archive_path, pet_root, item["expected_paths"])

    print("")
    print("Oxford-IIIT Pet download completed.")
    print(f"dataset root: {pet_root}")
    print("downloaded parts:")
    for part in selected_parts:
        print(f" - {part}")


if __name__ == "__main__":
    main()
