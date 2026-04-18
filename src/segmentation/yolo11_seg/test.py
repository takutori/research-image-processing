from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pycocotools import mask as mask_util

from src.segmentation.common.data import (
    ULTRALYTICS_CLS_TO_CATEGORY_ID,
    build_coco_gt,
    get_image_ids,
    get_image_path,
    resolve_coco_root,
)
from src.segmentation.common.evaluation import (
    compute_coco_mask_ap,
    polygon_to_rle,
    save_mask_ap_report,
    save_predictions_json,
)
from src.segmentation.yolo11_seg.config import load_experiment_config
from src.segmentation.yolo11_seg.model import build_yolo11_seg_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11-seg on COCO val2017 instance segmentation."
    )
    parser.add_argument(
        "--config",
        default="/workspace/config/segmentation/yolo11_seg_coco.yaml",
        help="Path to experiment config YAML.",
    )
    return parser.parse_args()


def run_inference(
    model,
    image_ids: list[int],
    images_dir: Path,
    coco_gt,
    img_size: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
) -> list[dict]:
    """Run YOLO11-seg predict and return COCO-format segmentation results."""
    image_paths = [str(get_image_path(coco_gt, iid, images_dir)) for iid in image_ids]
    dt_results: list[dict] = []

    for start in range(0, len(image_paths), batch):
        batch_paths = image_paths[start : start + batch]
        batch_ids = image_ids[start : start + batch]

        preds = model.predict(
            source=batch_paths,
            imgsz=img_size,
            device=device,
            verbose=False,
            conf=conf,
            iou=iou,
        )

        for result, image_id in zip(preds, batch_ids):
            if result.masks is None or len(result.masks) == 0:
                continue

            h, w = result.orig_shape
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for mask_xy, score, cls in zip(result.masks.xy, scores, classes):
                if len(mask_xy) < 3:
                    continue

                category_id = ULTRALYTICS_CLS_TO_CATEGORY_ID.get(int(cls), int(cls) + 1)
                polygon = mask_xy.flatten().tolist()
                rle = polygon_to_rle(polygon, h, w)

                dt_results.append({
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "segmentation": rle,
                    "score": round(float(score), 6),
                })

        if start % (batch * 10) == 0:
            print(f"  [{start}/{len(image_paths)}] processed")

    return dt_results


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    coco_root = resolve_coco_root()
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    images_dir = coco_root / "val2017"

    coco_gt = build_coco_gt(ann_file)
    image_ids = get_image_ids(coco_gt, max_images=config.test.max_images)

    print(f"Loading model: {config.test.model_name}")
    model = build_yolo11_seg_model(config.test.model_name)

    print(f"Running inference on {len(image_ids)} images ...")
    dt_results = run_inference(
        model=model,
        image_ids=image_ids,
        images_dir=images_dir,
        coco_gt=coco_gt,
        img_size=config.test.img_size,
        batch=config.test.batch,
        device=config.test.device,
        conf=config.test.conf,
        iou=config.test.iou,
    )

    print(f"Computing COCO mask AP ({len(dt_results)} masks) ...")
    ap_results = compute_coco_mask_ap(coco_gt, dt_results, image_ids=image_ids)

    output_dir = config.output_path / "best_test_results"
    save_mask_ap_report(output_dir, ap_results, "yolo11_seg", config.experiment_name)
    save_predictions_json(output_dir, dt_results)

    experiment_summary = {
        "experiment_name": config.experiment_name,
        "mode": "pretrained",
        "best_source": "pretrained",
        "best_run_name": config.test.model_name,
        "best_checkpoint_path": config.test.model_name,
        "best_valid_accuracy": ap_results["mask_ap50"],
        "test_accuracy": ap_results["mask_ap50"],
        "model_config": {
            "model_name": config.test.model_name,
            "img_size": config.test.img_size,
            "conf": config.test.conf,
            "iou": config.test.iou,
        },
        "test_mask_ap50": ap_results["mask_ap50"],
        "test_mask_ap": ap_results["mask_ap"],
    }
    summary_path = config.output_path / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(experiment_summary, ensure_ascii=False, indent=2))

    print(f"test_mask_ap50 : {ap_results['mask_ap50']:.4f}")
    print(f"test_mask_ap   : {ap_results['mask_ap']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
