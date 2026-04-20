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
    get_detection_lookup,
    get_image_ids,
    get_image_path,
    save_pose_report,
)
from src.pose_estimation.common.evaluation import save_pose_predictions_csv
from src.pose_estimation.grounding_dino_rtmpose.config import DEFAULT_CONFIG_PATH, load_experiment_config
from src.pose_estimation.grounding_dino_rtmpose.model import (
    build_grounding_dino_person_detector,
    generate_person_detections,
)
from src.pose_estimation.rtmpose.model import (
    build_rtmpose_model,
    pose_samples_to_coco_results,
    run_topdown_pose_inference,
)
from src.pose_estimation.rtmpose.test import build_gt_detection_lookup
from src.pose_estimation.common.evaluation import compute_coco_pose_ap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Grounding DINO + RTMPose on COCO person keypoints.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def evaluate_rtmpose_with_detection_lookup(
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
        pose_samples = run_topdown_pose_inference(model, get_image_path(coco_gt, image_id, images_dir), detections)
        pose_results, pose_rows = pose_samples_to_coco_results(image_id, pose_samples)
        dt_results.extend(pose_results)
        csv_rows.extend(pose_rows)
    ap_results = compute_coco_pose_ap(coco_gt, dt_results, image_ids=image_ids)
    return ap_results, dt_results, csv_rows


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    dataset_root = Path(config.dataset_root)
    ann_file = dataset_root / "annotations" / "person_keypoints_val2017.json"
    images_dir = dataset_root / "images" / "val2017"
    coco_gt = build_coco_gt(ann_file)
    image_ids = get_image_ids(coco_gt, max_images=config.test.max_images)

    rtmpose_model = build_rtmpose_model(
        config_path=config.test.rtmpose_config_path,
        checkpoint=config.test.rtmpose_checkpoint,
        device=config.test.device,
    )
    gt_lookup = build_gt_detection_lookup(coco_gt)
    print("Evaluating RTMPose with GT boxes ...")
    gt_ap_results, _, gt_csv_rows = evaluate_rtmpose_with_detection_lookup(
        model=rtmpose_model,
        coco_gt=coco_gt,
        images_dir=images_dir,
        image_ids=image_ids,
        detection_lookup=gt_lookup,
    )

    print("Running Grounding DINO person detection ...")
    processor, detector = build_grounding_dino_person_detector(
        model_name=config.test.detector_model,
        device=config.test.device,
    )
    gd_detections: list[dict] = []
    for image_id in image_ids:
        gd_detections.extend(
            generate_person_detections(
                processor=processor,
                model=detector,
                image_path=get_image_path(coco_gt, image_id, images_dir),
                image_id=image_id,
                box_threshold=config.test.box_threshold,
                text_threshold=config.test.text_threshold,
                crop_margin=config.test.crop_margin,
            )
        )
    gd_lookup = get_detection_lookup(gd_detections)
    print("Evaluating RTMPose with Grounding DINO boxes ...")
    gd_ap_results, _, gd_csv_rows = evaluate_rtmpose_with_detection_lookup(
        model=rtmpose_model,
        coco_gt=coco_gt,
        images_dir=images_dir,
        image_ids=image_ids,
        detection_lookup=gd_lookup,
    )

    output_dir = config.output_path / "best_test_results"
    gt_output_dir = output_dir / "gt_bbox"
    gd_output_dir = output_dir / "grounding_dino_bbox"
    save_pose_report(gt_output_dir, gt_ap_results, "rtmpose", config.experiment_name, extra_summary={"bbox_source": "ground_truth"})
    save_pose_predictions_csv(gt_output_dir, gt_csv_rows)
    save_pose_report(gd_output_dir, gd_ap_results, "grounding_dino_rtmpose", config.experiment_name, extra_summary={"bbox_source": "grounding_dino"})
    save_pose_predictions_csv(gd_output_dir, gd_csv_rows)

    comparison_summary = {
        "experiment_name": config.experiment_name,
        "gt_bbox_ap": gt_ap_results["ap"],
        "gt_bbox_ap50": gt_ap_results["ap50"],
        "grounding_dino_bbox_ap": gd_ap_results["ap"],
        "grounding_dino_bbox_ap50": gd_ap_results["ap50"],
        "detector_model": config.test.detector_model,
        "rtmpose_config_path": config.test.rtmpose_config_path,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2))
    (config.output_path / "experiment_summary.json").write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2))
    print(f"test completed. results saved to: {output_dir}")


if __name__ == "__main__":
    main()
