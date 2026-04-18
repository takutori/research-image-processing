from __future__ import annotations

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.segmentation.sam2.model import build_sam2_predictor


def build_grounding_dino(
    model_name: str,
    device: str,
) -> tuple:
    """Load Grounding DINO processor and model."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def chunk_class_names(
    class_names: list[str],
    chunk_size: int,
) -> list[tuple[list[str], list[int]]]:
    """Split class names into chunks with their original indices."""
    chunks = []
    for start in range(0, len(class_names), chunk_size):
        names = class_names[start : start + chunk_size]
        indices = list(range(start, start + len(names)))
        chunks.append((names, indices))
    return chunks


def build_text_prompt(class_names: list[str]) -> str:
    """Build a dot-separated text prompt from class names."""
    return ". ".join(class_names) + "."
