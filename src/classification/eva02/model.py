from __future__ import annotations

import timm
import torch.nn as nn


def build_eva02_model(
    num_classes: int,
    model_name: str = "eva02_base_patch14_224",
    pretrained: bool = True,
) -> nn.Module:
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

