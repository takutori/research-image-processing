# Dataset Download

## 方針

データセット本体は Git 管理せず、repo 外のホストディレクトリを使います。  
Docker コンテナ内では `/workspace/data` に mount されます。

- ホスト側: `HOST_DATA_DIR`
- コンテナ側: `/workspace/data`
- データ操作スクリプト: `src/manage_datasets/download_<dataset>.py`, `src/manage_datasets/delete_dataset.py`

例えば `HOST_DATA_DIR` の中では、以下のようにデータセットごとにフォルダを切って管理します。

```text
research-image-processing-data/
  coco/
  oxford_iiit_pet/
  visa/
  penn_fudan_pedestrian/
  mpii_human_pose/
```

## VisA

画像異常検知の最初の実験用としては、`MVTec AD` と並んで `VisA` が有力です。

- 工業系の異常検知データセット
- image-level / pixel-level のラベルあり
- `WinCLIP`, `PaDiM`, `PatchCore` などを比較しやすい

全量を取得する場合:

```bash
uv run python src/manage_datasets/download_visa.py
```

明示的に dataset 部分だけ指定する場合:

```bash
uv run python src/manage_datasets/download_visa.py --parts dataset
```

## Oxford-IIIT Pet

画像分類の最初の練習用としては、COCO より `Oxford-IIIT Pet` の方が扱いやすいです。

- 画像分類に素直な構造
- 品種ラベルがある
- セグメンテーション用 annotation も含まれる

全量を取得する場合:

```bash
uv run python src/manage_datasets/download_oxford_iiit_pet.py
```

必要な部分だけ取得する場合:

```bash
uv run python src/manage_datasets/download_oxford_iiit_pet.py --parts images
```
```bash
uv run python src/manage_datasets/download_oxford_iiit_pet.py --parts annotations
```

## COCO 2017

全量を取得する場合:

```bash
uv run python src/manage_datasets/download_coco.py
```

必要な split だけ取得する場合:

```bash
uv run python src/manage_datasets/download_coco.py --parts train annotations
```
```bash
uv run python src/manage_datasets/download_coco.py --parts val annotations
```
```bash
uv run python src/manage_datasets/download_coco.py --parts test image_info_test
```

## DTD (Describable Textures Dataset)

画像異常検知の tuning における合成異常生成（SWSA）のテクスチャソースとして使います。
anomalib の `PerlinAnomalyGenerator` が内部で参照します。

- 47 テクスチャカテゴリ × 120 枚 = 5,640 枚
- ライセンス: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 出典: https://www.robots.ox.ac.uk/~vgg/data/dtd/

全量を取得する場合:

```bash
uv run python src/manage_datasets/download_dtd.py
```

## COCO 2017 — YOLO11 用セットアップ

アノテーションダウンロード後、YOLO11 で学習する前に一度だけ実行します。

```bash
uv run python src/manage_datasets/setup_coco_yolo.py
```

実行内容:
- `coco/images/train2017` → `coco/train2017` のシンボリックリンク作成
- `coco/images/val2017` → `coco/val2017` のシンボリックリンク作成
- COCO JSON → YOLO 形式 `.txt` ラベルの変換 (`coco/labels/train2017/`, `coco/labels/val2017/`)
- `config/object_detection/coco_dataset.yaml` の `train` / `val` パスを更新

## COCO 2017 — YOLO11-pose 用セットアップ

person keypoints を使う `YOLO11-pose` は、検出用 `coco/` とは別に専用 root を作ります。

```bash
uv run python src/manage_datasets/setup_coco_yolo_pose.py
```

実行内容:
- `coco/images/train2017` / `val2017` から `coco_pose/images/train2017` / `val2017` を作成
- `person_keypoints_train2017.json` / `val2017.json` から YOLO pose 形式 `.txt` を生成
- `coco_pose/annotations/` に必要な JSON への symlink を作成
- `config/pose_estimation/coco_pose_dataset.yaml` を更新

## COCO 2017 — pose smoke subset

smoke では短時間で実装確認できるよう、小規模 subset を別 root に作ります。

```bash
uv run python src/manage_datasets/create_coco_pose_smoke.py
```

実行内容:
- `coco_pose_smoke/images/train2017` / `val2017` に subset 画像の symlink を作成
- `coco_pose_smoke/annotations/` に person keypoints / detection 用 subset JSON を作成
- `coco_pose_smoke/labels/train2017` / `val2017` に YOLO pose ラベルを生成
- `smoke_metadata.json` に subset 情報を保存

補足:
- `RTMPose` を使う場合は、dataset 準備とは別に `mmengine`, `mmcv-lite`, `xtcocotools`, `mmpose` を `uv pip install` で手動セットアップする
- 詳細な手順は `docs/train_test/pose_estimation.md` を参照
- 現段階ではスピード優先のため、姿勢推定は `YOLO11-pose` を主線に進める

## データセット削除

```bash
uv run python src/manage_datasets/delete_dataset.py
```

実行後、`HOST_DATA_DIR` 配下のデータセットフォルダ一覧から対象を選び、`Y` を入力した場合のみ削除します。
