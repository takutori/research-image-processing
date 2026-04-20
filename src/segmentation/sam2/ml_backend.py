from __future__ import annotations

import os

import cv2
import numpy as np
import requests
import torch
from io import BytesIO
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


CHECKPOINT_PATH = os.getenv(
    "SAM2_CHECKPOINT",
    "/workspace/checkpoints/segmentation/sam2/sam2.1_hiera_large.pt",
)
MODEL_CONFIG = os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SAM2Backend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading SAM 2 on {DEVICE} ...")
        sam2_model = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=DEVICE)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
        )
        print("SAM 2 ready.")

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            image = self._load_image(task)
            if image is None:
                results.append({"result": []})
                continue

            image_np = np.array(image.convert("RGB"))
            masks = self.mask_generator.generate(image_np)
            h, w = image_np.shape[:2]

            predictions = []
            for mask_data in masks:
                mask = mask_data["segmentation"].astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
                )
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) < 3:
                        continue
                    points = [
                        [float(p[0][0]) / w * 100, float(p[0][1]) / h * 100]
                        for p in approx
                    ]
                    predictions.append(
                        {
                            "from_name": "label",
                            "to_name": "image",
                            "type": "polygonlabels",
                            "value": {
                                "points": points,
                                "polygonlabels": ["object"],
                                "closed": True,
                            },
                            "score": float(mask_data["predicted_iou"]),
                        }
                    )

            results.append({"result": predictions})

        return results

    def _load_image(self, task: dict) -> Image.Image | None:
        image_url: str = task["data"].get("image", "")
        headers: dict = {}

        if image_url.startswith("/data"):
            full_url = f"{LABEL_STUDIO_URL}{image_url}"
            if LABEL_STUDIO_API_KEY:
                headers["Authorization"] = f"Token {LABEL_STUDIO_API_KEY}"
            response = requests.get(full_url, headers=headers)
            return Image.open(BytesIO(response.content))

        if image_url.startswith("http"):
            response = requests.get(image_url)
            return Image.open(BytesIO(response.content))

        if os.path.exists(image_url):
            return Image.open(image_url)

        print(f"Cannot load image: {image_url}")
        return None
