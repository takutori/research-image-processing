from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_detections(
    image: Image.Image | np.ndarray | Path | str,
    boxes: list[list[float]],
    labels: list[str],
    scores: list[float],
    score_threshold: float = 0.3,
    box_format: str = "xyxy",
) -> Image.Image:
    """Draw bounding boxes on an image.

    Args:
        image: PIL Image, numpy array (H, W, C), or path to image file.
        boxes: List of boxes. Format determined by box_format.
        labels: Class name for each box.
        scores: Confidence score for each box.
        score_threshold: Boxes below this score are skipped.
        box_format: "xyxy" (x1,y1,x2,y2) or "xywh" (x,y,w,h COCO format).

    Returns:
        PIL Image with detections drawn.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image.copy()

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    colors = _color_palette(len(set(labels)))
    label_color_map: dict[str, tuple[int, int, int]] = {}

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        if label not in label_color_map:
            label_color_map[label] = colors[len(label_color_map) % len(colors)]
        color = label_color_map[label]

        if box_format == "xywh":
            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{label} {score:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 16), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 16), text, fill="white", font=font)

    return image


def _color_palette(n: int) -> list[tuple[int, int, int]]:
    base = [
        (220, 50, 50), (50, 150, 220), (50, 200, 100), (230, 150, 30),
        (180, 80, 220), (30, 200, 200), (220, 120, 180), (100, 100, 220),
        (200, 200, 50), (50, 180, 150),
    ]
    palette = []
    for i in range(n):
        palette.append(base[i % len(base)])
    return palette
