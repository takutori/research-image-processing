from __future__ import annotations

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def build_grounding_dino_model(
    model_name: str = "IDEA-Research/grounding-dino-tiny",
    device: str = "cuda",
) -> tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    """Load Grounding DINO processor and model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device string.

    Returns:
        (processor, model) tuple. Model is moved to device and set to eval mode.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return processor, model


def build_text_prompt(class_names: list[str]) -> str:
    """Build a period-separated text prompt from a list of class names.

    Grounding DINO expects class names joined by '. '.
    Example: "person. bicycle. car."
    """
    return ". ".join(class_names) + "."


def chunk_class_names(
    class_names: list[str], chunk_size: int
) -> list[tuple[list[str], list[int]]]:
    """Split class names into chunks with their original indices.

    Returns:
        List of (chunk_names, chunk_indices) tuples.
        chunk_indices[i] is the index of chunk_names[i] in class_names.
    """
    chunks = []
    for start in range(0, len(class_names), chunk_size):
        names = class_names[start : start + chunk_size]
        indices = list(range(start, start + len(names)))
        chunks.append((names, indices))
    return chunks
