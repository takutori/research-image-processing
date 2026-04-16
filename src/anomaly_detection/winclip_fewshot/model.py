from __future__ import annotations

from pathlib import Path


def build_winclip_fewshot_model(
    *,
    class_name: str,
    scales: tuple = (2, 3),
    k_shot: int = 4,
):
    from anomalib.models import WinClip

    return WinClip(
        class_name=class_name,
        k_shot=k_shot,
        scales=tuple(scales),
        pre_processor=False,
    )
