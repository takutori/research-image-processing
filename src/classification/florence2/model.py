from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoProcessor, Florence2Model


def build_florence2_processor(model_name: str):
    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


class Florence2ForImageClassification(nn.Module):
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        self.backbone = Florence2Model.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        hidden_size = (
            self.backbone.config.vision_config.projection_dim
            if hasattr(self.backbone.config.vision_config, "projection_dim")
            else self.backbone.config.text_config.d_model
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.backbone.get_image_features(pixel_values=pixel_values)
        pooled = image_features.mean(dim=1)
        return self.classifier(pooled)


def build_florence2_model(num_classes: int, model_name: str = "microsoft/Florence-2-base-ft"):
    return Florence2ForImageClassification(model_name=model_name, num_classes=num_classes)
