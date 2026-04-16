from __future__ import annotations

from typing import Sequence


def build_stfpm_model(
    backbone: str = "resnet18",
    layers: Sequence[str] = ("layer1", "layer2", "layer3"),
    pre_trained: bool = True,
    image_size: int = 256,
):
    from anomalib.models import Stfpm
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Resize

    pre_processor = PreProcessor(Resize((image_size, image_size)))
    return Stfpm(
        backbone=backbone,
        layers=list(layers),
        pre_processor=pre_processor,
    )
