from __future__ import annotations


def build_fastflow_model(
    backbone: str = "resnet18",
    pre_trained: bool = True,
    flow_steps: int = 8,
    hidden_ratio: float = 1.0,
    image_size: int = 256,
):
    from anomalib.models import Fastflow
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Resize

    pre_processor = PreProcessor(Resize((image_size, image_size)))
    return Fastflow(
        backbone=backbone,
        pre_trained=pre_trained,
        flow_steps=flow_steps,
        hidden_ratio=hidden_ratio,
        pre_processor=pre_processor,
    )
