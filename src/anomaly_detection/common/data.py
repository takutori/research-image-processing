from __future__ import annotations

import csv
import os
from pathlib import Path

VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

DEFAULT_DATASET_ROOT = Path(os.environ.get("HOST_DATA_DIR", "")) / "visa" if os.environ.get("HOST_DATA_DIR") else Path(__file__).resolve().parents[4] / "data" / "visa"


def list_visa_categories(dataset_root: Path | str | None = None) -> list[str]:
    return list(VISA_CATEGORIES)


def setup_visa_pytorch(dataset_root: Path | str) -> None:
    """Build visa_pytorch/ symlink structure from split_csv/1cls.csv if not already done.

    anomalib's Visa datamodule expects:
        {dataset_root}/visa_pytorch/{category}/train/good/
        {dataset_root}/visa_pytorch/{category}/test/good/
        {dataset_root}/visa_pytorch/{category}/test/bad/
        {dataset_root}/visa_pytorch/{category}/ground_truth/bad/

    This mirrors what anomalib's apply_cls1_split() does but uses symlinks
    instead of copies to avoid duplicating disk usage.
    """
    dataset_root = Path(dataset_root)
    split_file = dataset_root / "split_csv" / "1cls.csv"
    visa_pytorch_root = dataset_root / "visa_pytorch"

    if visa_pytorch_root.exists():
        return

    if not split_file.exists():
        raise FileNotFoundError(f"VisA split CSV not found: {split_file}")

    print(f"setting up visa_pytorch/ symlinks at {visa_pytorch_root}")

    with split_file.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)

    for row in rows:
        category, split, label, image_path, mask_path = row
        label_dir = "good" if label == "normal" else "bad"

        img_src = dataset_root / image_path
        img_dst_dir = visa_pytorch_root / category / split / label_dir
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        img_dst = img_dst_dir / Path(image_path).name
        if not img_dst.exists() and not img_dst.is_symlink():
            img_dst.symlink_to(img_src)

        if split == "test" and label != "normal" and mask_path:
            msk_src = dataset_root / mask_path
            msk_dst_dir = visa_pytorch_root / category / "ground_truth" / label_dir
            msk_dst_dir.mkdir(parents=True, exist_ok=True)
            msk_dst = msk_dst_dir / Path(mask_path).name
            if not msk_dst.exists() and not msk_dst.is_symlink():
                msk_dst.symlink_to(msk_src)

    print("visa_pytorch/ setup complete")


def setup_dtd_symlink(dtd_data_root: Path | str, project_root: Path | str) -> None:
    """Create {project_root}/datasets/dtd -> {dtd_data_root} symlink.

    anomalib's PerlinAnomalyGenerator looks for DTD at './datasets/dtd'
    relative to the current working directory. This function creates that
    symlink so the generator can find the DTD textures.
    """
    dtd_data_root = Path(dtd_data_root)
    project_root = Path(project_root)
    datasets_dir = project_root / "datasets"
    dtd_link = datasets_dir / "dtd"

    if dtd_link.exists() or dtd_link.is_symlink():
        return

    dtd_images = dtd_data_root / "dtd"
    if not dtd_images.exists():
        print(f"warning: DTD not found at {dtd_images}. Run download_dtd.py first. Synthetic validation will fall back to Perlin-only noise.")
        return

    datasets_dir.mkdir(parents=True, exist_ok=True)
    dtd_link.symlink_to(dtd_images)
    print(f"created DTD symlink: {dtd_link} -> {dtd_images}")


def build_visa_datamodule(
    *,
    dataset_root: Path | str,
    category: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int | None = None,
    augmentations=None,
    train_augmentations=None,
    val_augmentations=None,
    test_augmentations=None,
    test_split_mode: str = "from_dir",
    test_split_ratio: float = 0.2,
    val_split_mode: str = "same_as_test",
    val_split_ratio: float = 0.5,
):
    """Build Visa datamodule with a workaround for anomalib 2.3.3 split enums.

    anomalib 2.3.3 passes enum values into VisaDataset, while VisaDataset only
    filters correctly when given plain strings ("train"/"test"). This subclass
    keeps the public datamodule behavior but forces string split names.
    """
    from anomalib.data import Visa
    from anomalib.data.datasets.image.visa import VisaDataset

    class PatchedVisa(Visa):
        def _setup(self, _stage: str | None = None) -> None:
            self.train_data = VisaDataset(
                split="train",
                root=self.split_root,
                category=self.category,
            )
            self.test_data = VisaDataset(
                split="test",
                root=self.split_root,
                category=self.category,
            )

    return PatchedVisa(
        root=dataset_root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        train_augmentations=train_augmentations,
        val_augmentations=val_augmentations,
        test_augmentations=test_augmentations,
        augmentations=augmentations,
        test_split_mode=test_split_mode,
        test_split_ratio=test_split_ratio,
        val_split_mode=val_split_mode,
        val_split_ratio=val_split_ratio,
        seed=seed,
    )
