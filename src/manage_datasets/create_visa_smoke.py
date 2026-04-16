from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lightweight VisA smoke subset with symlinks.")
    parser.add_argument("--train-normal", type=int, default=16, help="Number of normal train images per category.")
    parser.add_argument("--test-normal", type=int, default=8, help="Number of normal test images per category.")
    parser.add_argument("--test-anomaly", type=int, default=8, help="Number of anomalous test images per category.")
    parser.add_argument("--categories", nargs="+", default=["candle"], help="Target categories. Default: candle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset sampling.")
    return parser.parse_args()


def resolve_data_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    return Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root()
    source_root = data_root / "visa"
    smoke_root = data_root / "visa_smoke"

    split_csv_path = source_root / "split_csv" / "1cls.csv"
    if not split_csv_path.exists():
        raise FileNotFoundError(f"VisA split CSV not found: {split_csv_path}")

    subset_rows_by_csv = build_subset_rows_by_csv(
        source_root=source_root,
        categories=args.categories,
        train_normal=args.train_normal,
        test_normal=args.test_normal,
        test_anomaly=args.test_anomaly,
        seed=args.seed,
    )
    build_smoke_dataset(source_root=source_root, smoke_root=smoke_root, subset_rows_by_csv=subset_rows_by_csv)

    print("")
    print("VisA smoke subset created.")
    print(f"source root: {source_root}")
    print(f"smoke root: {smoke_root}")
    print(f"categories: {', '.join(args.categories)}")
    print(f"train normal per category: {args.train_normal}")
    print(f"test normal per category: {args.test_normal}")
    print(f"test anomaly per category: {args.test_anomaly}")


def load_rows(split_csv_path: Path) -> list[dict[str, str]]:
    with split_csv_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_subset_rows_by_csv(
    *,
    source_root: Path,
    categories: list[str],
    train_normal: int,
    test_normal: int,
    test_anomaly: int,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    subset_rows_by_csv: dict[str, list[dict[str, str]]] = {}
    for index, csv_name in enumerate(["1cls.csv", "2cls_fewshot.csv", "2cls_highshot.csv"]):
        rows = load_rows(source_root / "split_csv" / csv_name)
        subset_rows_by_csv[csv_name] = sample_rows(
            rows,
            categories=categories,
            train_normal=train_normal,
            test_normal=test_normal,
            test_anomaly=test_anomaly,
            seed=seed + index,
        )
    return subset_rows_by_csv


def sample_rows(
    rows: list[dict[str, str]],
    *,
    categories: list[str],
    train_normal: int,
    test_normal: int,
    test_anomaly: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    selected_rows: list[dict[str, str]] = []

    for category in categories:
        category_rows = [row for row in rows if row["object"] == category]
        train_normal_rows = [row for row in category_rows if row["split"] == "train" and row["label"] == "normal"]
        test_normal_rows = [row for row in category_rows if row["split"] == "test" and row["label"] == "normal"]
        test_anomaly_rows = [row for row in category_rows if row["split"] == "test" and row["label"] != "normal"]

        selected_rows.extend(_sample_exact(train_normal_rows, train_normal, rng, f"{category} train normal"))
        selected_rows.extend(_sample_exact(test_normal_rows, test_normal, rng, f"{category} test normal"))
        selected_rows.extend(_sample_exact(test_anomaly_rows, test_anomaly, rng, f"{category} test anomaly"))

    return selected_rows


def _sample_exact(rows: list[dict[str, str]], count: int, rng: random.Random, label: str) -> list[dict[str, str]]:
    if len(rows) < count:
        raise ValueError(f"Not enough samples for {label}: required {count}, found {len(rows)}")
    return sorted(rng.sample(rows, count), key=lambda row: row["image"])


def build_smoke_dataset(*, source_root: Path, smoke_root: Path, subset_rows_by_csv: dict[str, list[dict[str, str]]]) -> None:
    split_root = smoke_root / "split_csv"
    visa_pytorch_root = smoke_root / "visa_pytorch"
    raw_root = smoke_root
    metadata_path = smoke_root / "smoke_metadata.json"

    if smoke_root.exists():
        shutil.rmtree(smoke_root)

    split_root.mkdir(parents=True, exist_ok=True)
    visa_pytorch_root.mkdir(parents=True, exist_ok=True)

    all_rows = {
        (row["object"], row["split"], row["label"], row["image"], row["mask"])
        for rows in subset_rows_by_csv.values()
        for row in rows
    }

    for category, split, label, image_rel, mask_rel in sorted(all_rows):
        label_dir = "good" if label == "normal" else "bad"

        image_src = source_root / image_rel
        raw_image_dst = raw_root / image_rel
        raw_image_dst.parent.mkdir(parents=True, exist_ok=True)
        _symlink(image_src, raw_image_dst)

        image_dst_dir = visa_pytorch_root / category / split / label_dir
        image_dst_dir.mkdir(parents=True, exist_ok=True)
        image_dst = image_dst_dir / Path(image_rel).name
        _symlink(image_src, image_dst)

        if split == "test" and label != "normal" and mask_rel:
            mask_src = source_root / mask_rel
            raw_mask_dst = raw_root / mask_rel
            raw_mask_dst.parent.mkdir(parents=True, exist_ok=True)
            _symlink(mask_src, raw_mask_dst)

            mask_dst_dir = visa_pytorch_root / category / "ground_truth" / label_dir
            mask_dst_dir.mkdir(parents=True, exist_ok=True)
            mask_dst = mask_dst_dir / Path(mask_rel).name
            _symlink(mask_src, mask_dst)

    for csv_name, rows in subset_rows_by_csv.items():
        write_subset_csv(split_root / csv_name, rows)
    metadata = {csv_name: summarize_rows(rows) for csv_name, rows in subset_rows_by_csv.items()}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


def _symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(source)


def write_subset_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object", "split", "label", "image", "mask"])
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        category = row["object"]
        bucket = f'{row["split"]}_{"normal" if row["label"] == "normal" else "anomaly"}'
        summary.setdefault(category, {})
        summary[category][bucket] = summary[category].get(bucket, 0) + 1
    return summary


if __name__ == "__main__":
    main()
