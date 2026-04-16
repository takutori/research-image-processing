from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.common import list_visa_categories, save_json, setup_visa_pytorch
from src.anomaly_detection.winclip_fewshot.config import DEFAULT_CONFIG_PATH, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register WinCLIP few-shot reference images for VisA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def get_categories(config) -> list[str]:
    return config.categories or list_visa_categories()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    setup_visa_pytorch(config.dataset_root)

    reference_root = config.output_path / "reference_images"
    registered = register_reference_images(
        dataset_root=Path(config.dataset_root),
        reference_root=reference_root,
        split_csv_name=config.train.fewshot_split_csv,
        categories=get_categories(config),
        k_shot=config.model.k_shot,
    )
    summary = {
        "experiment_name": config.experiment_name,
        "mode": "none",
        "best_source": "train",
        "best_artifact_path": str(reference_root),
        "best_model_config": {
            "scales": config.model.scales,
            "k_shot": config.model.k_shot,
            "few_shot_source": str(reference_root),
        },
        "registered_reference_counts": registered,
        "test_image_level_metric": None,
        "test_pixel_level_metric": None,
    }
    save_json(config.output_path / "experiment_summary.json", summary)
    print(f"reference registration completed. saved to: {reference_root}")


def register_reference_images(
    *,
    dataset_root: Path,
    reference_root: Path,
    split_csv_name: str,
    categories: list[str],
    k_shot: int,
) -> dict[str, int]:
    split_path = dataset_root / "split_csv" / split_csv_name
    if not split_path.exists():
        raise FileNotFoundError(f"few-shot split CSV not found: {split_path}")

    if reference_root.exists():
        shutil.rmtree(reference_root)
    reference_root.mkdir(parents=True, exist_ok=True)
    category_set = set(categories)
    grouped_paths: dict[str, list[Path]] = {category: [] for category in categories}

    with split_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            for row in reader:
                category = row.get("category") or row.get("object")
                image_path = row.get("image_path") or row.get("image") or row.get("path")
                label = row.get("label", "normal")
                if category not in category_set or not image_path or label != "normal":
                    continue
                grouped_paths[category].append(dataset_root / image_path)
        else:
            handle.seek(0)
            reader2 = csv.reader(handle)
            next(reader2)
            for row in reader2:
                category, _, label, image_path, *_ = row
                if category not in category_set or label != "normal":
                    continue
                grouped_paths[category].append(dataset_root / image_path)

    registered: dict[str, int] = {}
    for category in categories:
        category_dir = reference_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        selected = grouped_paths.get(category, [])[:k_shot]
        if len(selected) < k_shot:
            raise ValueError(f"not enough normal few-shot images for {category}: required {k_shot}, found {len(selected)}")
        for image_path in selected:
            if not image_path.exists():
                raise FileNotFoundError(f"few-shot source image not found: {image_path}")
            target = category_dir / image_path.name
            target.symlink_to(image_path)
        registered[category] = len(selected)

    return registered


if __name__ == "__main__":
    main()
