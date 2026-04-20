from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.object_detection.grounding_dino.model import build_grounding_dino_model


def build_grounding_dino_person_detector(model_name: str, device: str):
    return build_grounding_dino_model(model_name=model_name, device=device)


def generate_person_detections(
    *,
    processor,
    model,
    image_path: Path,
    image_id: int,
    box_threshold: float,
    text_threshold: float,
    crop_margin: float,
) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    model_device = next(model.parameters()).device
    inputs = processor(images=image, text="person.", return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(height, width)],
    )[0]

    detections: list[dict] = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if "person" not in str(label).strip().lower():
            continue
        x1, y1, x2, y2 = box.tolist()
        if crop_margin > 0:
            margin_x = (x2 - x1) * crop_margin
            margin_y = (y2 - y1) * crop_margin
            x1 = max(0.0, x1 - margin_x)
            y1 = max(0.0, y1 - margin_y)
            x2 = min(float(width), x2 + margin_x)
            y2 = min(float(height), y2 + margin_y)
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return detections
