from __future__ import annotations


def build_winclip_zeroshot_model(class_name: str, scales: tuple = (2, 3)):
    from anomalib.models import WinClip

    return WinClip(
        class_name=class_name,
        k_shot=0,
        scales=tuple(scales),
    )
