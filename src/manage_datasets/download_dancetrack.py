from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


DANCETRACK_REPO_ID = "noahcao/dancetrack"
DANCETRACK_REPO_TYPE = "dataset"

DANCETRACK_DOWNLOADS = {
    "train": {
        "filenames": ["train1.zip", "train2.zip"],
    },
    "val": {
        "filenames": ["val.zip"],
    },
    "test": {
        "filenames": ["test1.zip", "test2.zip"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DanceTrack dataset files.")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["all"],
        help="Download target parts. Default: all",
    )
    return parser.parse_args()


def resolve_data_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    return Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"


def download_file(filename: str, destination: Path) -> None:
    if destination.exists():
        print(f"skip download: {destination.name}")
        return

    print(
        "start download: "
        f"https://huggingface.co/datasets/{DANCETRACK_REPO_ID}/resolve/main/{filename}"
    )
    print(f"downloading {destination.name} via huggingface_hub")
    cached_path = Path(
        hf_hub_download(
            repo_id=DANCETRACK_REPO_ID,
            filename=filename,
            repo_type=DANCETRACK_REPO_TYPE,
        )
    )
    shutil.copyfile(cached_path, destination)


def extraction_marker_path(archive_path: Path, destination_dir: Path) -> Path:
    markers_dir = destination_dir / ".extract_complete"
    markers_dir.mkdir(parents=True, exist_ok=True)
    return markers_dir / f"{archive_path.stem}.done"


def is_already_extracted(archive_path: Path, destination_dir: Path) -> bool:
    return extraction_marker_path(archive_path, destination_dir).exists()


def extract_zip(archive_path: Path, destination_dir: Path) -> None:
    if is_already_extracted(archive_path, destination_dir):
        print(f"skip extract: {archive_path.name}")
        return

    print(f"extracting: {archive_path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination_dir)
    extraction_marker_path(archive_path, destination_dir).write_text("", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root()
    dancetrack_root = data_root / "dancetrack"
    downloads_root = dancetrack_root / "_downloads"

    dancetrack_root.mkdir(parents=True, exist_ok=True)
    downloads_root.mkdir(parents=True, exist_ok=True)

    selected_parts = ["train", "val", "test"]
    if "all" not in args.parts:
        selected_parts = args.parts

    downloaded_archives: list[str] = []
    for part in selected_parts:
        item = DANCETRACK_DOWNLOADS[part]
        for filename in item["filenames"]:
            archive_path = downloads_root / filename
            download_file(filename, archive_path)
            extract_zip(archive_path, dancetrack_root)
            downloaded_archives.append(filename)

    print("")
    print("DanceTrack download completed.")
    print(f"dataset root: {dancetrack_root}")
    print("downloaded parts:")
    for part in selected_parts:
        print(f" - {part}")
    print("downloaded archives:")
    for filename in downloaded_archives:
        print(f" - {filename}")


if __name__ == "__main__":
    main()
