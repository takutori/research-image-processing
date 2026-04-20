from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchvision
from PIL import Image

from src.segmentation.common.data import (
    COCO_CLASS_NAMES,
    COCO_NAME_TO_ID,
    build_coco_gt,
    get_image_ids,
    get_image_path,
    resolve_coco_root,
)
from src.segmentation.common.evaluation import (
    binary_mask_to_rle,
    compute_coco_mask_ap,
    save_mask_ap_report,
    save_predictions_json,
)
from src.segmentation.grounded_sam.config import load_experiment_config
from src.segmentation.grounded_sam.model import (
    build_grounding_dino,
    build_text_prompt,
    chunk_class_names,
)
from src.segmentation.sam2.model import build_sam2_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Grounded-SAM on COCO val2017 instance segmentation."
    )
    parser.add_argument(
        "--config",
        default="/workspace/config/segmentation/grounded_sam_coco.yaml",
        help="Path to experiment config YAML.",
    )
    return parser.parse_args()


def _normalize_label(label_text: str) -> str:
    return " ".join(label_text.strip().lower().split())


def _apply_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Class-aware NMS across chunk detections for a single image."""
    if not detections:
        return detections

    kept: list[int] = []
    category_ids = sorted({int(d["category_id"]) for d in detections})

    for category_id in category_ids:
        indices = [i for i, d in enumerate(detections) if int(d["category_id"]) == category_id]
        boxes = torch.tensor(
            [
                [
                    detections[i]["bbox_xyxy"][0],
                    detections[i]["bbox_xyxy"][1],
                    detections[i]["bbox_xyxy"][2],
                    detections[i]["bbox_xyxy"][3],
                ]
                for i in indices
            ],
            dtype=torch.float32,
        )
        scores = torch.tensor([detections[i]["score"] for i in indices], dtype=torch.float32)
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        kept.extend(indices[k] for k in keep.tolist())

    kept.sort()
    return [detections[i] for i in kept]


def run_inference(
    gd_processor,
    gd_model,
    sam2_predictor,
    image_ids: list[int],
    images_dir: Path,
    coco_gt,
    box_threshold: float,
    text_threshold: float,
    chunk_size: int,
    device: str,
) -> list[dict]:
    """Run Grounding DINO → SAM 2 pipeline and return COCO segm format results."""
    chunks = chunk_class_names(COCO_CLASS_NAMES, chunk_size)
    dt_results: list[dict] = []

    for idx, image_id in enumerate(image_ids):
        if idx % 50 == 0:
            print(f"  [{idx}/{len(image_ids)}] image_id={image_id}")

        image_path = get_image_path(coco_gt, image_id, images_dir)
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        w, h = image_pil.size

        # --- Grounding DINO detection ---
        per_image_dets: list[dict] = []
        for chunk_names, _ in chunks:
            text_prompt = build_text_prompt(chunk_names)
            inputs = gd_processor(
                images=image_pil,
                text=text_prompt,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = gd_model(**inputs)

            results = gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(h, w)],
            )[0]

            for box, score, label_text in zip(
                results["boxes"], results["scores"], results["labels"]
            ):
                category_id = COCO_NAME_TO_ID.get(_normalize_label(str(label_text)))
                if category_id is None:
                    continue
                x1, y1, x2, y2 = box.tolist()
                per_image_dets.append({
                    "category_id": int(category_id),
                    "score": round(float(score), 6),
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

        per_image_dets = _apply_nms(per_image_dets, iou_threshold=0.5)
        if not per_image_dets:
            continue

        # --- SAM 2 box-prompted prediction ---
        sam2_predictor.set_image(image_np)
        boxes = np.array([d["bbox_xyxy"] for d in per_image_dets], dtype=np.float32)

        with torch.no_grad():
            masks, scores_sam, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )
        # masks: (N, 1, H, W) or (N, H, W) depending on multimask_output
        if masks.ndim == 4:
            masks = masks[:, 0]  # (N, H, W)

        for det, mask in zip(per_image_dets, masks):
            rle = binary_mask_to_rle(mask.astype(np.uint8))
            dt_results.append({
                "image_id": int(image_id),
                "category_id": det["category_id"],
                "segmentation": rle,
                "score": det["score"],
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

    print(f"Loading Grounding DINO: {config.test.detector_model}")
    gd_processor, gd_model = build_grounding_dino(
        model_name=config.test.detector_model,
        device=config.test.device,
    )

    print(f"Loading SAM 2: {Path(config.test.sam2_checkpoint).name}")
    sam2_predictor = build_sam2_predictor(
        checkpoint=config.test.sam2_checkpoint,
        model_config=config.test.sam2_config,
        device=config.test.device,
    )

    print(f"Running inference on {len(image_ids)} images ...")
    dt_results = run_inference(
        gd_processor=gd_processor,
        gd_model=gd_model,
        sam2_predictor=sam2_predictor,
        image_ids=image_ids,
        images_dir=images_dir,
        coco_gt=coco_gt,
        box_threshold=config.test.box_threshold,
        text_threshold=config.test.text_threshold,
        chunk_size=config.test.chunk_size,
        device=config.test.device,
    )

    print(f"Computing COCO mask AP ({len(dt_results)} masks) ...")
    ap_results = compute_coco_mask_ap(coco_gt, dt_results, image_ids=image_ids)

    output_dir = config.output_path / "best_test_results"
    save_mask_ap_report(output_dir, ap_results, "grounded_sam", config.experiment_name)
    save_predictions_json(output_dir, dt_results)

    experiment_summary = {
        "experiment_name": config.experiment_name,
        "mode": "zeroshot",
        "best_source": "zeroshot",
        "best_run_name": "grounded_sam",
        "best_checkpoint_path": config.test.sam2_checkpoint,
        "best_valid_accuracy": ap_results["mask_ap50"],
        "test_accuracy": ap_results["mask_ap50"],
        "model_config": {
            "detector_model": config.test.detector_model,
            "sam2_checkpoint": config.test.sam2_checkpoint,
            "box_threshold": config.test.box_threshold,
            "text_threshold": config.test.text_threshold,
            "chunk_size": config.test.chunk_size,
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
