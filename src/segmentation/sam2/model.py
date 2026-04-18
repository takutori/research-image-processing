from __future__ import annotations

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_sam2_mask_generator(
    checkpoint: str,
    model_config: str,
    device: str,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.92,
) -> SAM2AutomaticMaskGenerator:
    """Build SAM 2 automatic mask generator."""
    sam2_model = build_sam2(model_config, checkpoint, device=device)
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )


def build_sam2_predictor(
    checkpoint: str,
    model_config: str,
    device: str,
) -> SAM2ImagePredictor:
    """Build SAM 2 image predictor (for box-prompted mode, e.g. Grounded-SAM)."""
    sam2_model = build_sam2(model_config, checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)
