# research-image-processing

画像処理・コンピュータビジョン案件を受けられる状態まで到達するための学習・調査・実装用ワークスペースです。

## Docker 開発環境

このリポジトリは `uv` を前提に、Docker 上で Python 環境を動かします。

### 初回ビルド

GPU を使う前提の標準起動:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

CPU のみで起動したい場合:

```bash
TORCH_VARIANT=cpu docker compose build
TORCH_VARIANT=cpu docker compose up -d
```

データセット保存先は repo 外のホストディレクトリを使います。起動前に `.env` を用意してください。

```bash
cp .env.example .env
```

`.env` の `HOST_DATA_DIR` を自分の環境に合わせて更新します。

### コンテナに入る

```bash
docker compose exec workspace bash
```

### ノートブックを起動する

```bash
docker compose exec workspace uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

ブラウザから `http://localhost:8888` を開きます。

### VS Code から notebook を使う

VS Code 上で `.ipynb` を開きつつ、実行環境だけ Docker コンテナ内 Jupyter に向けることもできます。

1. コンテナを起動する
2. コンテナ内で Jupyter Lab を起動する
3. 起動ログに出る `http://localhost:8888/?token=...` の URL を控える
4. VS Code で notebook を開く
5. コマンドパレットで `Jupyter: Select Notebook Kernel` を開く
6. `Existing Jupyter Server` を選ぶ
7. 上の token 付き URL を入力する
8. コンテナ側の kernel を選ぶ

この接続で `torch.cuda.is_available()` が `True` なら、VS Code 上の notebook からコンテナ内 GPU 環境を使えています。

### データ配置

データセット本体は Git 管理せず、repo 外のホストディレクトリを `/workspace/data` に mount します。

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

COCO 2017 全量を取得する場合:

```bash
uv run python src/manage_datasets/download_coco.py
```

必要な split だけ取得する場合:

```bash
uv run python src/manage_datasets/download_coco.py --parts train annotations
uv run python src/manage_datasets/download_coco.py --parts val annotations
uv run python src/manage_datasets/download_coco.py --parts test image_info_test
```

データセットを削除する場合:

```bash
uv run python src/manage_datasets/delete_dataset.py
```

## 依存関係

- Python 3.11
- `uv` による依存管理
- `PyTorch`, `torchvision`
- `OpenCV`, `albumentations`
- `jupyterlab`, 可視化・集計系ライブラリ

## GPU 利用

NVIDIA GPU を前提にしています。ホスト側で以下が必要です。

- NVIDIA Driver
- Docker
- NVIDIA Container Toolkit または Docker Desktop の GPU 連携

この構成では、再現性を優先して以下に固定しています。

- `torch==2.6.0`
- `torchvision==0.21.0`
- GPU wheel: `cu124`
- Python: `3.11`

そのため、ホスト側 NVIDIA Driver は CUDA 12.4 系 wheel を実行できるだけの新しさが必要です。

起動後の確認:

```bash
docker compose exec workspace uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

通常依存は `uv sync` で入り、`torch` / `torchvision` だけは build 時と起動時に固定 version で `uv pip install` します。CPU のみで運用したいときは `TORCH_VARIANT=cpu` を指定してください。

補足:

- `uv run` 時の自動同期は無効化しています
- CPU 用と GPU 用で `.venv` volume を分けています

そのため、CPU と GPU を切り替えても毎回 `torch` 系を入れ直しにくい構成です。
