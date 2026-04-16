from __future__ import annotations


def build_patchcore_model(
    backbone: str = "wide_resnet50_2",
    layers: list[str] | None = None,
    pre_trained: bool = True,
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
    image_size: int = 224,
):
    from anomalib.models import Patchcore
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Resize

    if layers is None:
        layers = ["layer2", "layer3"]
    pre_processor = PreProcessor(Resize((image_size, image_size)))
    return Patchcore(
        backbone=backbone,
        layers=layers,
        pre_trained=pre_trained,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        pre_processor=pre_processor,
    )
