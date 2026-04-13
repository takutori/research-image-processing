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
  penn_fudan_pedestrian/
  mpii_human_pose/
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

## データセット削除

```bash
uv run python src/manage_datasets/delete_dataset.py
```

実行後、`HOST_DATA_DIR` 配下のデータセットフォルダ一覧から対象を選び、`Y` を入力した場合のみ削除します。
