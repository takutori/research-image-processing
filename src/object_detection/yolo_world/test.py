from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.object_detection.common.data import (
    COCO_CLASS_NAMES,
    ULTRALYTICS_CLS_TO_CATEGORY_ID,
    build_coco_gt,
    get_image_ids,
    get_image_path,
    resolve_coco_root,
)
from src.object_detection.common.evaluation import (
    compute_coco_map,
    save_detection_report,
    save_predictions_json,
)
from src.object_detection.yolo_world.config import load_experiment_config
from src.object_detection.yolo_world.model import build_yolo_world_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-World on COCO val2017.")
    parser.add_argument(
        "--config",
        default="/workspace/config/object_detection/yolo_world_coco.yaml",
    )
    return parser.parse_args()


def predict_coco_val(
    model,
    coco_gt,
    images_dir: Path,
    image_ids: list[int],
    img_size: int,
    batch_size: int,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
) -> list[dict]:
    """Run YOLO-World predict on COCO val2017 images."""
    image_paths = [str(get_image_path(coco_gt, iid, images_dir)) for iid in image_ids]

    dt_results: list[dict] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_ids = image_ids[start : start + batch_size]
        if start % 500 == 0:
            print(f"  [{start}/{len(image_paths)}]")

        preds = model.predict(
            source=batch_paths,
            imgsz=img_size,
            device=device,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
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

    coco_root = resolve_coco_root()
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    images_dir = coco_root / "val2017"

    coco_gt = build_coco_gt(ann_file)
    image_ids = get_image_ids(coco_gt, max_images=config.test.max_images)

    print(f"Loading model: {config.test.model_name}")
    model = build_yolo_world_model(config.test.model_name)
    model.set_classes(COCO_CLASS_NAMES)

    print(f"Running predict on {len(image_ids)} images ...")
    dt_results = predict_coco_val(
        model=model,
        coco_gt=coco_gt,
        images_dir=images_dir,
        image_ids=image_ids,
        img_size=config.test.img_size,
        batch_size=config.test.batch_size,
        conf_threshold=config.test.conf_threshold,
        iou_threshold=config.test.iou_threshold,
        device=config.test.device,
    )

    print(f"Computing COCO mAP ({len(dt_results)} detections) ...")
    map_results = compute_coco_map(coco_gt, dt_results, image_ids=image_ids)

    output_dir = config.output_path / "best_test_results"
    save_detection_report(output_dir, map_results, "yolo_world", config.experiment_name)
    save_predictions_json(output_dir, dt_results)

    summary = {
        "experiment_name": config.experiment_name,
        "mode": "zeroshot",
        "best_source": "zeroshot",
        "best_run_name": "zeroshot",
        "best_checkpoint_path": config.test.model_name,
        "best_valid_map50": map_results["map50"],
        "best_valid_map": map_results["map"],
        "test_map50": map_results["map50"],
        "test_map": map_results["map"],
        "model_config": {
            "model_name": config.test.model_name,
            "img_size": config.test.img_size,
            "conf_threshold": config.test.conf_threshold,
        },
    }
    summary_path = config.output_path / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"test_map50 : {map_results['map50']:.4f}")
    print(f"test_map   : {map_results['map']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
