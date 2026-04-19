from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def require_mmpose() -> tuple:
    try:
        from mmpose.apis import inference_topdown, init_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RTMPose requires mmpose, mmengine, mmcv-lite, and xtcocotools at runtime. "
            "Install the MMPose runtime stack before running pose_estimation/rtmpose scripts."
        ) from exc
    return init_model, inference_topdown


def resolve_mmpose_config_path(config_path: str) -> str:
    candidate = Path(config_path)
    if candidate.exists():
        return str(candidate)

    try:
        import mmpose
    except ModuleNotFoundError:
        return config_path

    package_root = Path(mmpose.__file__).resolve().parent
    search_roots = [
        package_root,
        package_root / ".mim",
        package_root / ".mim" / "configs",
        package_root.parent,
    ]
    for root in search_roots:
        resolved = root / config_path
        if resolved.exists():
            return str(resolved)
    return config_path


def build_rtmpose_model(config_path: str, checkpoint: str | None, device: str):
    init_model, _ = require_mmpose()
    return init_model(resolve_mmpose_config_path(config_path), checkpoint, device=device)


def run_topdown_pose_inference(model, image_path: Path, detections: list[dict] | None = None) -> list:
    _, inference_topdown = require_mmpose()
    if detections:
        boxes_xyxy = np.array(
            [
                [
                    float(det["bbox"][0]),
                    float(det["bbox"][1]),
                    float(det["bbox"][0] + det["bbox"][2]),
                    float(det["bbox"][1] + det["bbox"][3]),
                ]
                for det in detections
            ],
            dtype=np.float32,
        )
    else:
        boxes_xyxy = None
    return inference_topdown(model, str(image_path), bboxes=boxes_xyxy, bbox_format="xyxy")


def pose_samples_to_coco_results(image_id: int, pose_samples: list) -> tuple[list[dict], list[dict]]:
    from src.pose_estimation.common.data import flatten_keypoints_with_scores

    dt_results: list[dict] = []
    csv_rows: list[dict] = []

    for sample_index, sample in enumerate(pose_samples):
        pred_instances = getattr(sample, "pred_instances", None)
        if pred_instances is None:
            continue

        keypoints = _to_numpy(getattr(pred_instances, "keypoints", None))
        keypoint_scores = _to_numpy(getattr(pred_instances, "keypoint_scores", None))
        bboxes = _to_numpy(getattr(pred_instances, "bboxes", None))
        bbox_scores = _to_numpy(getattr(pred_instances, "bbox_scores", None))

        if keypoints is None:
            continue
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if keypoint_scores is None:
            keypoint_scores = np.ones(keypoints.shape[:2], dtype=np.float32)
        elif keypoint_scores.ndim == 1:
            keypoint_scores = keypoint_scores[None, ...]
        if bboxes is None:
            bboxes = np.zeros((keypoints.shape[0], 4), dtype=np.float32)
        if bbox_scores is None:
            bbox_scores = np.ones((keypoints.shape[0],), dtype=np.float32)

        for person_index in range(keypoints.shape[0]):
            person_keypoints = keypoints[person_index]
            person_scores = keypoint_scores[person_index]
            x1, y1, x2, y2 = [float(value) for value in bboxes[person_index].tolist()]
            bbox_score = float(bbox_scores[person_index])
            flat_keypoints = flatten_keypoints_with_scores(
                person_keypoints[:, :2].tolist(),
                person_scores.tolist(),
            )
            pose_score = float(bbox_score * max(0.0, float(person_scores.mean())))
            bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
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
                    "instance_id": f"{image_id}_{sample_index}_{person_index}",
                    "score": pose_score,
                    "bbox_x": bbox_xywh[0],
                    "bbox_y": bbox_xywh[1],
                    "bbox_w": bbox_xywh[2],
                    "bbox_h": bbox_xywh[3],
                    "num_keypoints": int(len(person_scores)),
                    "mean_keypoint_score": float(person_scores.mean()),
                    "keypoints": str(flat_keypoints),
                }
            )
    return dt_results, csv_rows
