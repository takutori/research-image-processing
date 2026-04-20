from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.pose_estimation.common import (
    build_coco_gt,
    compute_coco_pose_ap,
    get_image_ids,
    get_image_path,
    save_pose_predictions_csv,
    save_pose_report,
)
from src.pose_estimation.common.data import flatten_keypoints_with_scores
from src.pose_estimation.yolo11_pose.config import DEFAULT_CONFIG_PATH, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO11-pose on COCO person keypoints.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def predict_pose(
    checkpoint_path: str | Path,
    dataset_root: Path,
    img_size: int,
    batch: int,
    device: str,
    max_images: int | None,
) -> tuple[list[dict], list[dict]]:
    ann_file = dataset_root / "annotations" / "person_keypoints_val2017.json"
    images_dir = dataset_root / "images" / "val2017"
    coco_gt = build_coco_gt(ann_file)
    image_ids = get_image_ids(coco_gt, max_images=max_images)
    image_paths = [str(get_image_path(coco_gt, image_id, images_dir)) for image_id in image_ids]

    model = YOLO(str(checkpoint_path))
    dt_results: list[dict] = []
    csv_rows: list[dict] = []

    for start in range(0, len(image_paths), batch):
        batch_paths = image_paths[start : start + batch]
        batch_ids = image_ids[start : start + batch]
        predictions = model.predict(
            source=batch_paths,
            imgsz=img_size,
            device=device,
            verbose=False,
            conf=0.001,
        )
        for result, image_id in zip(predictions, batch_ids):
            if result.keypoints is None or result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            box_scores = result.boxes.conf.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy()
            for instance_index, (box, bbox_score, keypoints_array) in enumerate(zip(boxes, box_scores, keypoints)):
                x1, y1, x2, y2 = box.tolist()
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                flat_keypoints = flatten_keypoints_with_scores(
                    keypoints_array[:, :2].tolist(),
                    keypoints_array[:, 2].tolist(),
                )
                pose_score = float(float(bbox_score) * max(0.0, float(keypoints_array[:, 2].mean())))
                dt_results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": 1,
                        "keypoints": flat_keypoints,
                        "score": pose_score,
                        "bbox": bbox_xywh,
                    }
                )
                csv_rows.append(
                    {
                        "image_id": int(image_id),
                        "instance_id": f"{image_id}_{instance_index}",
                        "score": pose_score,
                        "bbox_x": bbox_xywh[0],
                        "bbox_y": bbox_xywh[1],
                        "bbox_w": bbox_xywh[2],
                        "bbox_h": bbox_xywh[3],
                        "num_keypoints": int(keypoints_array.shape[0]),
                        "mean_keypoint_score": float(keypoints_array[:, 2].mean()),
                        "keypoints": str(flat_keypoints),
                    }
                )
    return dt_results, csv_rows


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    summary_path = config.output_path / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"experiment summary not found: {summary_path}. Run train.py first.")

    summary = json.loads(summary_path.read_text())
    dataset_root = Path(config.dataset_root)
    ann_file = dataset_root / "annotations" / "person_keypoints_val2017.json"
    coco_gt = build_coco_gt(ann_file)

    dt_results, csv_rows = predict_pose(
        checkpoint_path=summary["best_checkpoint_path"],
        dataset_root=dataset_root,
        img_size=int(summary["best_model_config"]["img_size"]),
        batch=config.train.batch,
        device=args.device or config.train.device,
        max_images=args.max_images or config.test.max_images,
    )
    ap_results = compute_coco_pose_ap(
        coco_gt,
        dt_results,
        image_ids=get_image_ids(coco_gt, max_images=args.max_images or config.test.max_images),
    )

    output_dir = config.output_path / "best_test_results"
    save_pose_report(output_dir, ap_results, "yolo11_pose", config.experiment_name)
    save_pose_predictions_csv(output_dir, csv_rows)

    summary["test_ap"] = ap_results["ap"]
    summary["test_ap50"] = ap_results["ap50"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"test completed. results saved to: {output_dir}")


if __name__ == "__main__":
    main()
