from __future__ import annotations


def build_padim_model(
    backbone: str = "resnet18",
    layers: list[str] | None = None,
    pre_trained: bool = True,
    n_features: int | None = None,
    image_size: int = 224,
):
    from anomalib.models import Padim
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Resize

    if layers is None:
        layers = ["layer1", "layer2", "layer3"]
    pre_processor = PreProcessor(Resize((image_size, image_size)))
    return Padim(
        backbone=backbone,
        layers=layers,
        pre_trained=pre_trained,
        n_features=n_features,
        pre_processor=pre_processor,
    )
