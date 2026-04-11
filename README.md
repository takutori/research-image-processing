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

### コンテナに入る

```bash
docker compose exec workspace bash
```

### ノートブックを起動する

```bash
docker compose exec workspace uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

ブラウザから `http://localhost:8888` を開きます。

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
