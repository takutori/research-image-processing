from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.common import (
    build_coco_gt,
    compute_coco_pose_ap,
    get_detection_lookup,
    get_image_ids,
    get_image_path,
    group_annotations_by_image,
    save_pose_predictions_csv,
    save_pose_report,
)
from src.pose_estimation.rtmpose.config import DEFAULT_CONFIG_PATH, load_experiment_config
from src.pose_estimation.rtmpose.model import (
    build_rtmpose_model,
    pose_samples_to_coco_results,
    run_topdown_pose_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RTMPose on COCO person keypoints.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--bbox-file", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def load_detections(path: Path) -> dict[int, list[dict]]:
    detections = json.loads(path.read_text())
    person_detections = [det for det in detections if int(det.get("category_id", -1)) == 1]
    return get_detection_lookup(person_detections)


def build_gt_detection_lookup(coco_gt) -> dict[int, list[dict]]:
    annotations = coco_gt.dataset["annotations"]
    grouped = group_annotations_by_image(annotations)
    return {
        image_id: [
            {
                "image_id": image_id,
                "category_id": 1,
                "bbox": annotation["bbox"],
                "score": 1.0,
            }
            for annotation in rows
            if int(annotation.get("iscrowd", 0)) == 0 and int(annotation.get("num_keypoints", 0)) > 0
        ]
        for image_id, rows in grouped.items()
    }


def evaluate_pose_model(
    *,
    model,
    coco_gt,
    images_dir: Path,
    image_ids: list[int],
    detection_lookup: dict[int, list[dict]],
) -> tuple[dict[str, float], list[dict], list[dict]]:
    dt_results: list[dict] = []
    csv_rows: list[dict] = []
    for index, image_id in enumerate(image_ids):
        if index % 100 == 0:
            print(f"  [{index}/{len(image_ids)}] image_id={image_id}")
        detections = detection_lookup.get(image_id, [])
        if not detections:
            continue
        image_path = get_image_path(coco_gt, image_id, images_dir)
        pose_samples = run_topdown_pose_inference(model, image_path, detections)
        pose_results, pose_rows = pose_samples_to_coco_results(image_id, pose_samples)
        dt_results.extend(pose_results)
        csv_rows.extend(pose_rows)
    ap_results = compute_coco_pose_ap(coco_gt, dt_results, image_ids=image_ids)
    return ap_results, dt_results, csv_rows


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    summary_path = config.output_path / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"experiment summary not found: {summary_path}. Run train.py first.")

    summary = json.loads(summary_path.read_text())
    checkpoint_path = summary["best_checkpoint_path"]
    config_path = summary["best_model_config"]["config_path"]
    dataset_root = Path(config.dataset_root)
    ann_file = dataset_root / "annotations" / "person_keypoints_val2017.json"
    images_dir = dataset_root / "images" / "val2017"

    coco_gt = build_coco_gt(ann_file)
    image_ids = get_image_ids(coco_gt, max_images=args.max_images or config.test.max_images)
    bbox_file = args.bbox_file or config.test.bbox_file
    if bbox_file:
        detection_lookup = load_detections(Path(bbox_file))
    else:
        detection_lookup = build_gt_detection_lookup(coco_gt)

    model = build_rtmpose_model(config_path=config_path, checkpoint=checkpoint_path, device=config.test.device)
    ap_results, _, csv_rows = evaluate_pose_model(
        model=model,
        coco_gt=coco_gt,
        images_dir=images_dir,
        image_ids=image_ids,
        detection_lookup=detection_lookup,
    )

    output_dir = config.output_path / "best_test_results"
    save_pose_report(
        output_dir,
        ap_results,
        "rtmpose",
        config.experiment_name,
        extra_summary={"bbox_source": Path(bbox_file).name if bbox_file else "ground_truth"},
    )
    save_pose_predictions_csv(output_dir, csv_rows)

    summary["test_ap"] = ap_results["ap"]
    summary["test_ap50"] = ap_results["ap50"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"test completed. results saved to: {output_dir}")


if __name__ == "__main__":
    main()
