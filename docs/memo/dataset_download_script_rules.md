# Dataset Download Script Rules

このメモは、`src/manage_datasets/download_<dataset>.py` を新規追加・更新するときのルールです。

## 読むタイミング

- データセット download 用の `.py` ファイルを作るとき

## 基本方針

- データセット本体は Git 管理しない
- 保存先は `HOST_DATA_DIR` を優先し、未設定時だけ `project_root / "data"` を使う
- スクリプトは `src/manage_datasets` 配下にフラットに置く
- ファイル名は `download_<dataset>.py`

## 実装ルール

- 既存の downloader と同じインターフェースに揃える
  - `parse_args()`
  - `resolve_data_root()`
  - `download_file()`
  - `main()`
- 取得対象が複数ある場合は `--parts` を持たせる
  - 例: `images`, `annotations`, `dataset`, `all`
- ダウンロード済みファイルは再取得しない
  - `destination.exists()` なら `skip download`
- 展開済みなら再展開しない
  - `expected_paths` で判定する
- 進捗は既存スクリプトと同じ形式で出す
  - `start download: ...`
  - `downloading ...`
  - `extracting: ...`
  - `skip download: ...`
  - `skip extract: ...`

## 展開ルール

- 圧縮形式に応じて適切な標準ライブラリを使う
  - `.zip` -> `zipfile`
  - `.tar`, `.tar.gz` -> `tarfile`
- 展開先は `<dataset_root>/`
- ダウンロードファイルは `<dataset_root>/_downloads/` に置く

## URL とライセンス

- できるだけ公式配布元を使う
- 公式 URL が不安定な場合は、その理由が明確なときだけ代替 URL を使う
- 学習用に公開 repo へ置いてよいかは、データ本体ではなくスクリプトだけを commit する前提で判断する
- README / docs には、必要なら出典 URL と用途を短く書く

## docs 更新

- `docs/setup/download_dataset.md` に追加方法を書く
- 既存の記法に合わせる
  - データセット概要
  - 全量取得コマンド
  - 必要なら部分取得コマンド

## 確認ルール

- 少なくとも `python3 -m compileall` で構文確認する
- 実ダウンロードは、ユーザーが明示的に求めたときだけ実行する

## 避けること

- データ本体を repo 配下へ commit する
- downloader ごとにバラバラな CLI を作る
- 既存 downloader と違う保存先規約にする
