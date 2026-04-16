from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


VISA_DOWNLOADS = {
    "dataset": {
        "filename": "VisA_20220922.tar",
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VisA anomaly detection dataset files.")
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
    with tarfile.open(archive_path, mode="r") as archive:
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
    with tarfile.open(archive_path, mode="r") as archive:
        archive.extractall(destination_dir)
    return expected_paths


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root()
    visa_root = data_root / "visa"
    downloads_root = visa_root / "_downloads"

    visa_root.mkdir(parents=True, exist_ok=True)
    downloads_root.mkdir(parents=True, exist_ok=True)

    selected_parts = ["dataset"]
    if "all" not in args.parts:
        selected_parts = args.parts

    extracted_paths: dict[str, list[str]] = {}
    for part in selected_parts:
        item = VISA_DOWNLOADS[part]
        archive_path = downloads_root / item["filename"]
        download_file(item["url"], archive_path)
        extracted_paths[part] = extract_archive(archive_path, visa_root)

    print("")
    print("VisA download completed.")
    print(f"dataset root: {visa_root}")
    print("downloaded parts:")
    for part in selected_parts:
        print(f" - {part}")
    print("top-level extracted paths:")
    for part in selected_parts:
        for extracted_path in extracted_paths[part]:
            print(f"   - {extracted_path}")


if __name__ == "__main__":
    main()
