from __future__ import annotations

import timm
import torch.nn as nn


def build_convnext_v2_model(
    num_classes: int,
    model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
    pretrained: bool = True,
) -> nn.Module:
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
