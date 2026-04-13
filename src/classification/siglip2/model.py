from __future__ import annotations

from transformers import AutoImageProcessor, AutoModelForImageClassification


def build_siglip2_processor(model_name: str):
    return AutoImageProcessor.from_pretrained(model_name)


def build_siglip2_model(
    num_classes: int,
    model_name: str = "google/siglip2-base-patch16-224",
):
    return AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

