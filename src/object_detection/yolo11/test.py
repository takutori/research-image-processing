from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.object_detection.common.data import (
    build_coco_gt,
    get_image_ids,
    get_image_path,
    resolve_coco_root,
    ULTRALYTICS_CLS_TO_CATEGORY_ID,
)
from src.object_detection.common.evaluation import (
    compute_coco_map,
    save_detection_report,
    save_predictions_json,
)
from src.object_detection.yolo11.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 on COCO val2017.")
    parser.add_argument(
        "--config",
        default="/workspace/config/object_detection/yolo11_coco.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of val images (for smoke/quick runs).",
    )
    return parser.parse_args()


def predict_coco_val(
    checkpoint_path: str | Path,
    coco_root: Path,
    img_size: int,
    batch: int,
    device: str,
    max_images: int | None,
) -> list[dict]:
    """Run YOLO11 predict on COCO val2017 and return COCO-format results."""
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    images_dir = coco_root / "val2017"

    coco_gt = build_coco_gt(ann_file)
    image_ids = get_image_ids(coco_gt, max_images=max_images)

    image_paths = [str(get_image_path(coco_gt, iid, images_dir)) for iid in image_ids]

    model = YOLO(str(checkpoint_path))
    dt_results: list[dict] = []

    # Process in batches to avoid OOM.
    # result.path may be renamed by ultralytics (e.g. "image0.jpg"), so track
    # image_ids by batch position instead of by path.
    for start in range(0, len(image_paths), batch):
        batch_paths = image_paths[start : start + batch]
        batch_ids = image_ids[start : start + batch]
        preds = model.predict(
            source=batch_paths,
            imgsz=img_size,
            device=device,
            verbose=False,
            conf=0.001,   # low threshold to capture all detections for mAP
            iou=0.6,
        )
        for result, image_id in zip(preds, batch_ids):
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls in zip(boxes_xyxy, scores, classes):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                category_id = ULTRALYTICS_CLS_TO_CATEGORY_ID.get(cls, cls + 1)
                dt_results.append({
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(float(score), 6),
                })

    return dt_results


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    summary_path = config.output_path / "experiment_summary.json"
    summary = json.loads(summary_path.read_text())

    best_checkpoint_path = Path(summary["best_checkpoint_path"])
    img_size = summary["best_model_config"]["img_size"]
    device = args.device or config.train.device

    coco_root = resolve_coco_root()
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    coco_gt = build_coco_gt(ann_file)

    print(f"Running predict on COCO val2017 with {best_checkpoint_path.name} ...")
    dt_results = predict_coco_val(
        checkpoint_path=best_checkpoint_path,
        coco_root=coco_root,
        img_size=img_size,
        batch=config.train.batch,
        device=device,
        max_images=args.max_images,
    )

    print("Computing COCO mAP ...")
    map_results = compute_coco_map(coco_gt, dt_results)

    output_dir = config.output_path / "best_test_results"
    save_detection_report(output_dir, map_results, "yolo11", config.experiment_name)
    save_predictions_json(output_dir, dt_results)

    summary["test_map50"] = map_results["map50"]
    summary["test_map"] = map_results["map"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"test_map50 : {map_results['map50']:.4f}")
    print(f"test_map   : {map_results['map']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
