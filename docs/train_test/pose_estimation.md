# Pose Estimation

当面は `YOLO11-pose` を主線にして進めます。
`RTMPose` と `Grounding DINO + RTMPose` は OpenMMLab 専用環境を切ってから再開します。

比較候補としては次の 3 系統を維持します。

- `RTMPose`
- `YOLO11-pose`
- `Grounding DINO + RTMPose`

## Dataset Preparation

フルデータ用:

```bash
uv run python src/manage_datasets/setup_coco_yolo_pose.py
```

smoke 用 subset:

```bash
uv run python src/manage_datasets/create_coco_pose_smoke.py
```

## Current Priority

- 今すぐ進める: `YOLO11-pose`
- 先に整備するもの: `coco_pose`, `coco_pose_smoke`
- 専用環境待ちで保留: `RTMPose`, `Grounding DINO + RTMPose`

## RTMPose Runtime Dependency

`RTMPose` は `uv sync` の通常依存には含めず、軽量な `mmcv-lite` を使って別 install します。
この環境では `mmcv` full package が source build に落ちやすいため、まずは ops なし構成で通します。
以下 4 つは `uv pip install` で手動セットアップする前提です。

```bash
uv pip install --upgrade pip setuptools wheel
uv pip install mmengine mmcv-lite
uv pip install cython
uv pip install --no-binary xtcocotools --force-reinstall xtcocotools
uv pip install --no-deps mmpose
```

## RTMPose

現時点では保留です。OpenMMLab 専用環境を用意してから再開します。

full:

```bash
uv run python src/pose_estimation/rtmpose/train.py --config /workspace/config/pose_estimation/rtmpose_coco.yaml
uv run python src/pose_estimation/rtmpose/test.py --config /workspace/config/pose_estimation/rtmpose_coco.yaml
```

smoke:

```bash
uv run python src/pose_estimation/rtmpose/train.py --config /workspace/config/pose_estimation/rtmpose_coco_smoke.yaml
uv run python src/pose_estimation/rtmpose/test.py --config /workspace/config/pose_estimation/rtmpose_coco_smoke.yaml
```

## YOLO11-pose

full:

```bash
uv run python src/pose_estimation/yolo11_pose/train.py --config /workspace/config/pose_estimation/yolo11_pose_coco.yaml
uv run python src/pose_estimation/yolo11_pose/test.py --config /workspace/config/pose_estimation/yolo11_pose_coco.yaml
```

現在の full config は、スピード優先で `yolo11n-pose` 単体・`batch=8`・`workers=4` の保守的な設定にしています。
`CUDA error: unknown error` が出る場合は、まずこの設定のまま再実行し、それでも落ちるなら `CUDA_LAUNCH_BLOCKING=1` を付けて切り分けます。

smoke:

```bash
uv run python src/pose_estimation/yolo11_pose/train.py --config /workspace/config/pose_estimation/yolo11_pose_coco_smoke.yaml
uv run python src/pose_estimation/yolo11_pose/test.py --config /workspace/config/pose_estimation/yolo11_pose_coco_smoke.yaml
```

## Grounding DINO + RTMPose

現時点では保留です。`RTMPose` の専用環境が整ってから再開します。

full:

```bash
uv run python src/pose_estimation/grounding_dino_rtmpose/test.py --config /workspace/config/pose_estimation/grounding_dino_rtmpose_coco.yaml
```

smoke:

```bash
uv run python src/pose_estimation/grounding_dino_rtmpose/test.py --config /workspace/config/pose_estimation/grounding_dino_rtmpose_coco_smoke.yaml
```

## Notes

- `xtcocotools` は wheel だと `numpy` ABI 不整合を踏むことがあるため、source build で入れる前提です
- `RTMPose` 周辺の依存は現時点では `pyproject.toml` ではなく `uv pip install` で入れる例外運用です
- スピード優先のため、現段階では `YOLO11-pose` の実験を先に進めます
- `Grounding DINO + RTMPose` は `GT bbox` と `Grounding DINO bbox` の 2 条件を同時に出力します
- smoke は実装確認用であり、性能比較の結論には使いません
