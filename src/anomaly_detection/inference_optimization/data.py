from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.anomaly_detection.common import build_visa_datamodule, setup_visa_pytorch


@dataclass
class PreparedBatch:
    images: torch.Tensor
    image_paths: list[str | None]
    labels: np.ndarray
    masks: list[np.ndarray | None]
    categories: list[str]

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])


@dataclass(frozen=True)
class DatasetSpec:
    dataset_root: str
    category: str
    image_size: int
    batch_size: int
    num_workers: int
    seed: int


def build_test_dataloader(spec: DatasetSpec):
    from torchvision.transforms.v2 import Resize

    resize = Resize((spec.image_size, spec.image_size))
    datamodule = build_visa_datamodule(
        dataset_root=spec.dataset_root,
        category=spec.category,
        train_batch_size=spec.batch_size,
        eval_batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        seed=spec.seed,
        augmentations=resize,
    )
    datamodule.setup()
    return datamodule.test_dataloader()


def prepare_test_batches(
    *,
    dataset_root: str,
    categories: Iterable[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    max_batches: int | None = None,
) -> list[PreparedBatch]:
    """Load test samples once so timing excludes dataloader/decode/resize work."""
    setup_visa_pytorch(Path(dataset_root))
    prepared: list[PreparedBatch] = []
    for category in categories:
        dataloader = build_test_dataloader(
            DatasetSpec(
                dataset_root=dataset_root,
                category=category,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed,
            )
        )
        for batch in dataloader:
            images = _as_plain_tensor(_get_field(batch, "image")).to(device=device, non_blocking=True)
            labels = _labels_to_numpy(_get_field(batch, "gt_label", _get_field(batch, "label")))
            masks = _masks_to_list(_get_field(batch, "gt_mask", _get_field(batch, "mask", None)), batch_size=int(images.shape[0]))
            image_paths = _paths_to_list(_get_field(batch, "image_path", None), batch_size=int(images.shape[0]))
            prepared.append(
                PreparedBatch(
                    images=images,
                    image_paths=image_paths,
                    labels=labels,
                    masks=masks,
                    categories=[category] * int(images.shape[0]),
                )
            )
            if max_batches is not None and len(prepared) >= max_batches:
                return prepared
    if not prepared:
        raise RuntimeError("No test batches were prepared. Check dataset_root/categories/config.")
    return prepared


def _get_field(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_plain_tensor(value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if type(value) is torch.Tensor:
        return value
    return value.as_subclass(torch.Tensor)


def _labels_to_numpy(value) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.int64)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).astype(np.int64).reshape(-1)


def _masks_to_list(value, *, batch_size: int) -> list[np.ndarray | None]:
    if value is None:
        return [None] * batch_size
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.shape[0] != batch_size:
        return [arr.squeeze()] + [None] * max(0, batch_size - 1)
    return [arr[i].squeeze() for i in range(batch_size)]


def _paths_to_list(value, *, batch_size: int) -> list[str | None]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, (list, tuple)):
        paths = [str(v) for v in value]
    else:
        paths = [str(value)]
    if len(paths) < batch_size:
        paths.extend([None] * (batch_size - len(paths)))
    return paths[:batch_size]
