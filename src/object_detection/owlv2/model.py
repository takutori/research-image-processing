from __future__ import annotations

import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor


def build_owlv2_model(
    model_name: str = "google/owlv2-base-patch16-ensemble",
    device: str = "cuda",
) -> tuple[Owlv2Processor, Owlv2ForObjectDetection]:
    """Load OWLv2 processor and model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device string.

    Returns:
        (processor, model) tuple. Model is moved to device and set to eval mode.
    """
    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return processor, model
