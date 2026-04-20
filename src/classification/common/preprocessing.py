from __future__ import annotations

from typing import Sequence

from torchvision import transforms


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

OPENCLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
OPENCLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

SIGLIP_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)


def build_supervised_train_transform(
    image_size: int = 224,
    scale: tuple[float, float] = (0.7, 1.0),
    ratio: tuple[float, float] = (0.8, 1.25),
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_supervised_eval_transform(
    image_size: int = 224,
    resize_size: int = 256,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_openclip_eval_transform(image_size: int = 224) -> transforms.Compose:
    return build_supervised_eval_transform(
        image_size=image_size,
        resize_size=image_size,
        mean=OPENCLIP_MEAN,
        std=OPENCLIP_STD,
    )


def build_siglip_eval_transform(image_size: int = 224) -> transforms.Compose:
    return build_supervised_eval_transform(
        image_size=image_size,
        resize_size=image_size,
        mean=SIGLIP_MEAN,
        std=SIGLIP_STD,
    )
