from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


DTD_DOWNLOADS = {
    "dataset": {
        "filename": "dtd-r1.0.1.tar.gz",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DTD (Describable Textures Dataset) files.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["dataset", "all"],
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


def list_expected_paths(archive_path: Path) -> list[str]:
    top_level_paths: set[str] = set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            normalized = Path(member.name)
            if not normalized.parts:
                continue
            top_level_paths.add(normalized.parts[0])
    return sorted(top_level_paths)


def is_already_extracted(destination_dir: Path, expected_paths: list[str]) -> bool:
    return all((destination_dir / expected_path).exists() for expected_path in expected_paths)


def extract_archive(archive_path: Path, destination_dir: Path) -> list[str]:
    expected_paths = list_expected_paths(archive_path)
    if is_already_extracted(destination_dir, expected_paths):
        print(f"skip extract: {archive_path.name}")
        return expected_paths

    print(f"extracting: {archive_path.name}")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(destination_dir)
    return expected_paths


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root()
    dtd_root = data_root / "dtd"
    downloads_root = dtd_root / "_downloads"

    dtd_root.mkdir(parents=True, exist_ok=True)
    downloads_root.mkdir(parents=True, exist_ok=True)

    selected_parts = ["dataset"]
    if "all" not in args.parts:
        selected_parts = args.parts

    extracted_paths: dict[str, list[str]] = {}
    for part in selected_parts:
        item = DTD_DOWNLOADS[part]
        archive_path = downloads_root / item["filename"]
        download_file(item["url"], archive_path)
        extracted_paths[part] = extract_archive(archive_path, dtd_root)

    print("")
    print("DTD download completed.")
    print(f"dataset root: {dtd_root}")
    print("downloaded parts:")
    for part in selected_parts:
        print(f" - {part}")
    print("top-level extracted paths:")
    for part in selected_parts:
        for extracted_path in extracted_paths[part]:
            print(f"   - {extracted_path}")


if __name__ == "__main__":
    main()
