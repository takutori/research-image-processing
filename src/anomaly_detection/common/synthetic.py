from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SyntheticSplitSummary:
    category: str
    num_train_normal_images: int
    num_val_images: int


def build_visa_synthetic_datamodule(
    *,
    dataset_root: str,
    category: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split_ratio: float,
    seed: int,
):
    """Build VisA datamodule with explicit synthetic validation behavior.

    This keeps the repository-side contract aligned with the rewrite design:
    training uses the normal-only train split, then anomalib performs the
    hold-out split and wraps the validation portion with SyntheticAnomalyDataset.
    """
    from torchvision.transforms.v2 import Resize
    from src.anomaly_detection.common.data import build_visa_datamodule

    resize = Resize((image_size, image_size))
    return build_visa_datamodule(
        dataset_root=dataset_root,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        augmentations=resize,
        val_split_mode="synthetic",
        val_split_ratio=val_split_ratio,
        seed=seed,
    )


def summarize_synthetic_split(datamodule, *, category: str) -> SyntheticSplitSummary:
    """Return train/validation sizes after anomalib creates the synthetic split."""
    train_data = getattr(datamodule, "train_data", None)
    val_data = getattr(datamodule, "val_data", None)
    return SyntheticSplitSummary(
        category=category,
        num_train_normal_images=len(train_data) if train_data is not None else 0,
        num_val_images=len(val_data) if val_data is not None else 0,
    )
