from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from src.segmentation.common.data import (
    build_coco_gt,
    get_image_ids,
    get_image_path,
    resolve_coco_root,
)
from src.segmentation.common.evaluation import (
    compute_recall_and_iou,
    save_sam2_report,
)
from src.segmentation.sam2.config import load_experiment_config
from src.segmentation.sam2.model import build_sam2_mask_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM 2 automatic mode on COCO val2017."
    )
    parser.add_argument(
        "--config",
        default="/workspace/config/segmentation/sam2_coco.yaml",
        help="Path to experiment config YAML.",
    )
    return parser.parse_args()


def run_inference(
    mask_generator,
    image_ids: list[int],
    images_dir: Path,
    coco_gt,
) -> dict[int, list[np.ndarray]]:
    """Run SAM 2 automatic mask generation and return masks per image_id."""
    masks_per_image: dict[int, list[np.ndarray]] = {}

    for idx, image_id in enumerate(image_ids):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(image_ids)}] image_id={image_id}")

        image_path = get_image_path(coco_gt, image_id, images_dir)
        image_np = np.array(Image.open(image_path).convert("RGB"))

        mask_data = mask_generator.generate(image_np)
        masks_per_image[image_id] = [m["segmentation"] for m in mask_data]

    return masks_per_image


def build_per_image_rows(
    coco_gt,
    masks_per_image: dict[int, list[np.ndarray]],
    image_ids: list[int],
) -> list[dict]:
    """Build per-image summary rows for predictions.csv."""
    from pycocotools import mask as mask_util

    rows = []
    for image_id in image_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=image_id, iscrowd=False)
        n_gt = len(ann_ids)
        n_pred = len(masks_per_image.get(image_id, []))
        rows.append({
            "image_id": image_id,
            "n_gt": n_gt,
            "n_pred": n_pred,
        })
    return rows


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    coco_root = resolve_coco_root()
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    images_dir = coco_root / "val2017"

    coco_gt = build_coco_gt(ann_file)
    image_ids = get_image_ids(coco_gt, max_images=config.test.max_images)

    print(f"Loading SAM 2 model: {Path(config.test.checkpoint).name}")
    mask_generator = build_sam2_mask_generator(
        checkpoint=config.test.checkpoint,
        model_config=config.test.model_config,
        device=config.test.device,
        points_per_side=config.test.points_per_side,
        pred_iou_thresh=config.test.pred_iou_thresh,
        stability_score_thresh=config.test.stability_score_thresh,
    )

    print(f"Running SAM 2 automatic mask generation on {len(image_ids)} images ...")
    masks_per_image = run_inference(
        mask_generator=mask_generator,
        image_ids=image_ids,
        images_dir=images_dir,
        coco_gt=coco_gt,
    )

    print("Computing recall and best-match IoU ...")
    recall_results = compute_recall_and_iou(
        coco_gt=coco_gt,
        sam2_masks_per_image=masks_per_image,
        image_ids=image_ids,
        iou_threshold=0.5,
    )

    per_image_rows = build_per_image_rows(coco_gt, masks_per_image, image_ids)

    output_dir = config.output_path / "best_test_results"
    save_sam2_report(
        output_dir=output_dir,
        recall_results=recall_results,
        per_image_rows=per_image_rows,
        model_name="sam2",
        experiment_name=config.experiment_name,
    )

    experiment_summary = {
        "experiment_name": config.experiment_name,
        "mode": "zeroshot",
        "best_source": "zeroshot",
        "best_run_name": Path(config.test.checkpoint).name,
        "best_checkpoint_path": config.test.checkpoint,
        "best_valid_accuracy": recall_results.get("recall_50", 0.0),
        "test_accuracy": recall_results.get("recall_50", 0.0),
        "model_config": {
            "checkpoint": config.test.checkpoint,
            "points_per_side": config.test.points_per_side,
            "pred_iou_thresh": config.test.pred_iou_thresh,
            "stability_score_thresh": config.test.stability_score_thresh,
        },
        "test_recall_50": recall_results.get("recall_50", 0.0),
        "test_mean_best_iou": recall_results["mean_best_iou"],
    }
    summary_path = config.output_path / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(experiment_summary, ensure_ascii=False, indent=2))

    print(f"recall@50      : {recall_results.get('recall_50', 0.0):.4f}")
    print(f"mean_best_iou  : {recall_results['mean_best_iou']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
