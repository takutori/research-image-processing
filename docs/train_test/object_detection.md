# Object Detection Train / Test

このドキュメントは、**すでに Docker コンテナに入っていて、作業ディレクトリが `/workspace` である** 前提で書いています。

## 前提

- `COCO 2017` (train2017 + val2017 + annotations) をダウンロード済み
- `uv sync` 済み
- 学習 / テストの実行はユーザーが行う

## 設定ファイル

train / test は、リポジトリ直下の `config/object_detection/*.yaml` を指定して実行します。

現在の例:

```bash
config/object_detection/yolo11_coco.yaml
config/object_detection/yolo11_coco_smoke.yaml
config/object_detection/grounding_dino_coco.yaml
config/object_detection/grounding_dino_coco_smoke.yaml
config/object_detection/yolo_world_coco.yaml
config/object_detection/yolo_world_coco_smoke.yaml
config/object_detection/owlv2_coco.yaml
config/object_detection/owlv2_coco_smoke.yaml
```

## 共通方針

- `YOLO11` は supervised (学習あり)、`Grounding DINO` / `YOLO-World` / `OWLv2` は zero-shot
- 評価指標はすべて `pycocotools` で統一 (mAP@[0.5:0.95], mAP@0.5)
- `train.py` は学習 / チューニングだけ (`YOLO11` のみ)
- `test.py` は評価のみ。`YOLO11` は `experiment_summary.json` から最良 checkpoint を自動解決する
- smoke は `max_images=50` (zero-shot) または `fraction=0.002` (YOLO11) で実行時間を短縮する
- smoke でも artifact 形式は本番と同一

## YOLO11

supervised モデル。`tuning_mode: grid` で複数モデル・複数画像サイズを自動的に探索し、最良 checkpoint を記録します。

### 1. train 実行

```bash
uv run python src/object_detection/yolo11/train.py \
  --config config/object_detection/yolo11_coco.yaml
```

`tuning_mode` に従って動きます。

- `none`: 単発学習
- `grid`: `yolo11n / yolo11s / yolo11m` × `img_size 640 / 1280` の grid search

今の `yolo11_coco.yaml` は `grid` なので、**train 実行後に最良 checkpoint が `experiment_summary.json` に記録されます。**

主な出力:

```
checkpoints/object_detection/yolo11/yolo11_coco/
  experiment_summary.json
  grid/
    trial_001/ ... trial_006/
    tuning_results.csv
```

### 2. test 実行

```bash
uv run python src/object_detection/yolo11/test.py \
  --config config/object_detection/yolo11_coco.yaml
```

`experiment_summary.json` から最良 checkpoint を読み込み、COCO val2017 で評価します。

主な出力:

```
checkpoints/object_detection/yolo11/yolo11_coco/
  best_test_results/
    summary.json            # mAP@[0.5:0.95], mAP@0.5, mAP@0.75, map_small/medium/large
    per_category_ap.csv     # カテゴリ別 AP50, AP
    predictions.json        # COCO results format
```

### smoke

```bash
# smoke: epochs=1, yolo11n.pt, img_size=640, fraction=0.002 (≈236 枚)
uv run python src/object_detection/yolo11/train.py \
  --config config/object_detection/yolo11_coco_smoke.yaml

uv run python src/object_detection/yolo11/test.py \
  --config config/object_detection/yolo11_coco_smoke.yaml
```

## Grounding DINO

zero-shot モデル。`train.py` はなく、`test.py` だけで実験全体が完結します。

80 クラス名を 20 クラスずつ chunk に分割してプロンプトに渡し、chunk 間で NMS を適用します。

### test 実行

```bash
uv run python src/object_detection/grounding_dino/test.py \
  --config config/object_detection/grounding_dino_coco.yaml
```

主な出力:

```
checkpoints/object_detection/grounding_dino/grounding_dino_coco/
  experiment_summary.json
  best_test_results/
    summary.json
    per_category_ap.csv
    predictions.json
```

### smoke

```bash
# smoke: tiny モデル, max_images=50
uv run python src/object_detection/grounding_dino/test.py \
  --config config/object_detection/grounding_dino_coco_smoke.yaml
```

## YOLO-World

zero-shot モデル。`model.set_classes()` で COCO 80 クラスをテキストクエリとして登録し推論します。`train.py` はありません。

### test 実行

```bash
uv run python src/object_detection/yolo_world/test.py \
  --config config/object_detection/yolo_world_coco.yaml
```

主な出力:

```
checkpoints/object_detection/yolo_world/yolo_world_coco/
  experiment_summary.json
  best_test_results/
    summary.json
    per_category_ap.csv
    predictions.json
```

### smoke

```bash
# smoke: yolov8s-worldv2.pt, max_images=50
uv run python src/object_detection/yolo_world/test.py \
  --config config/object_detection/yolo_world_coco_smoke.yaml
```

## OWLv2

zero-shot モデル。各クラスを `"a photo of a {label}"` のテキストクエリとして渡します。`train.py` はありません。

### test 実行

```bash
uv run python src/object_detection/owlv2/test.py \
  --config config/object_detection/owlv2_coco.yaml
```

主な出力:

```
checkpoints/object_detection/owlv2/owlv2_coco/
  experiment_summary.json
  best_test_results/
    summary.json
    per_category_ap.csv
    predictions.json
```

### smoke

```bash
# smoke: owlv2-base-patch16-ensemble, max_images=50
uv run python src/object_detection/owlv2/test.py \
  --config config/object_detection/owlv2_coco_smoke.yaml
```

## 評価 notebook

全モデルの結果を横断比較する場合はこれを使います。

```bash
notebooks/object_detection/03_evaluation.ipynb
```

この notebook では以下を比較できます。

- mAP@0.5 / mAP@[0.5:0.95] の全モデル比較
- オブジェクトサイズ別 mAP (small / medium / large)
- カテゴリ別 AP@0.5 (上位 20 カテゴリ)
- YOLO11 の grid tuning 結果一覧
- Supervised vs Zero-shot のまとめ

## experiment_summary.json フォーマット

```json
{
  "experiment_name": "yolo11_coco",
  "mode": "grid",
  "best_source": "grid",
  "best_run_name": "grid/trial_003",
  "best_checkpoint_path": "/workspace/checkpoints/object_detection/yolo11/yolo11_coco/grid/trial_003/weights/best.pt",
  "best_valid_map50": 0.55,
  "test_map50": null,
  "test_map": null
}
```

zero-shot モデルは `best_source: "zeroshot"` / `best_valid_map50 == test_map50`。

## 注意

- `YOLO11` / `YOLO-World` は `ultralytics` 依存です。
- `Grounding DINO` / `OWLv2` は `transformers` 依存です。
- 学習前に `uv sync` を実行してください。
- COCO 2017 の category_id は 1-90 (非連続) です。`ultralytics` 内部インデックス (0-79) との変換は `src/object_detection/common/data.py` の `ULTRALYTICS_CLS_TO_CATEGORY_ID` で管理しています。
