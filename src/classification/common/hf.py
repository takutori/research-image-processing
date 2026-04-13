from __future__ import annotations

from collections.abc import Callable

import torch


def build_hf_processor_collate_fn(
    processor,
    *,
    prompt_text: str | None = None,
) -> Callable[[list[dict[str, object]]], dict[str, object]]:
    def collate_fn(samples: list[dict[str, object]]) -> dict[str, object]:
        images = [sample["image"] for sample in samples]
        if prompt_text is None:
            encoded = processor(images=images, return_tensors="pt")
        else:
            encoded = processor(
                text=[prompt_text] * len(samples),
                images=images,
                return_tensors="pt",
                padding=True,
            )

        return {
            "image": encoded["pixel_values"],
            "model_inputs": encoded,
            "label": torch.tensor([int(sample["label"]) for sample in samples], dtype=torch.long),
            "image_path": [str(sample["image_path"]) for sample in samples],
            "image_stem": [str(sample["image_stem"]) for sample in samples],
            "breed_name": [str(sample["breed_name"]) for sample in samples],
        }

    return collate_fn
