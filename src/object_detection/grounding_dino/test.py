from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from src.object_detection.common.data import (
    COCO_CLASS_NAMES,
    COCO_NAME_TO_ID,
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
from src.object_detection.grounding_dino.config import load_experiment_config
from src.object_detection.grounding_dino.model import (
    build_grounding_dino_model,
    build_text_prompt,
    chunk_class_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Grounding DINO on COCO val2017.")
    parser.add_argument(
        "--config",
        default="/workspace/config/object_detection/grounding_dino_coco.yaml",
    )
    return parser.parse_args()


def _apply_nms_across_chunks(
    detections: list[dict], iou_threshold: float = 0.5
) -> list[dict]:
    """Apply class-aware NMS to merge detections from multiple text-prompt chunks."""
    if not detections:
        return detections

    try:
        import torchvision
    except ImportError:
        return detections

    kept_indices: list[int] = []
    category_ids = sorted({int(d["category_id"]) for d in detections})

    for category_id in category_ids:
        category_indices = [
            idx for idx, det in enumerate(detections)
            if int(det["category_id"]) == category_id
        ]
        boxes = torch.tensor(
            [
                [
                    detections[idx]["bbox"][0],
                    detections[idx]["bbox"][1],
                    detections[idx]["bbox"][0] + detections[idx]["bbox"][2],
                    detections[idx]["bbox"][1] + detections[idx]["bbox"][3],
                ]
                for idx in category_indices
            ],
            dtype=torch.float32,
        )
        scores = torch.tensor(
            [detections[idx]["score"] for idx in category_indices],
            dtype=torch.float32,
        )
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        kept_indices.extend(category_indices[i] for i in keep.tolist())

    kept_indices.sort()
    return [detections[i] for i in kept_indices]


def _normalize_label(label_text: str) -> str:
    return " ".join(label_text.strip().lower().split())


def _resolve_category_id(label_text: str) -> int | None:
    label = _normalize_label(label_text)
    return COCO_NAME_TO_ID.get(label)


def run_inference(
    processor,
    model,
    image_ids: list[int],
    images_dir: Path,
    coco_gt,
    box_threshold: float,
    text_threshold: float,
    chunk_size: int,
    device: str,
) -> list[dict]:
    """Run Grounding DINO inference with chunked text prompts."""
    chunks = chunk_class_names(COCO_CLASS_NAMES, chunk_size)
    dt_results: list[dict] = []

    for idx, image_id in enumerate(image_ids):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(image_ids)}] image_id={image_id}")

        image_path = get_image_path(coco_gt, image_id, images_dir)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        per_image_dets: list[dict] = []

        for chunk_names, _chunk_indices in chunks:
            text_prompt = build_text_prompt(chunk_names)

            inputs = processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(h, w)],
            )[0]

            for box, score, label_text in zip(
                results["boxes"], results["scores"], results["labels"]
            ):
                category_id = _resolve_category_id(str(label_text))
                if category_id is None:
                    continue

                x1, y1, x2, y2 = box.tolist()
                coco_w, coco_h = x2 - x1, y2 - y1
                per_image_dets.append({
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [round(x1, 2), round(y1, 2), round(coco_w, 2), round(coco_h, 2)],
                    "score": round(float(score), 6),
                })

        # NMS across chunks for this image
        per_image_dets = _apply_nms_across_chunks(per_image_dets, iou_threshold=0.5)
        dt_results.extend(per_image_dets)

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
    processor, model = build_grounding_dino_model(
        model_name=config.test.model_name,
        device=config.test.device,
    )

    print(f"Running inference on {len(image_ids)} images (chunk_size={config.test.chunk_size}) ...")
    dt_results = run_inference(
        processor=processor,
        model=model,
        image_ids=image_ids,
        images_dir=images_dir,
        coco_gt=coco_gt,
        box_threshold=config.test.box_threshold,
        text_threshold=config.test.text_threshold,
        chunk_size=config.test.chunk_size,
        device=config.test.device,
    )

    print(f"Computing COCO mAP ({len(dt_results)} detections) ...")
    map_results = compute_coco_map(coco_gt, dt_results, image_ids=image_ids)

    output_dir = config.output_path / "best_test_results"
    save_detection_report(output_dir, map_results, "grounding_dino", config.experiment_name)
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
            "box_threshold": config.test.box_threshold,
            "text_threshold": config.test.text_threshold,
            "chunk_size": config.test.chunk_size,
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
