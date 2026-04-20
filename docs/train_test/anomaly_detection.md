# Anomaly Detection Train / Test

このドキュメントは、**すでに Docker コンテナに入っていて、作業ディレクトリが `/workspace` である** 前提で書いています。

## 前提

- `VisA` をダウンロード済み
- `uv sync` 済み
- `anomalib` を追加した後の環境同期も済み
- 学習 / テストの実行はユーザーが行う

## 設定ファイル

train / test は、リポジトリ直下の `config/anomaly_detection/*.yaml` を指定して実行します。

現在の例:

```bash
config/anomaly_detection/padim_visa.yaml
config/anomaly_detection/padim_visa_smoke.yaml
config/anomaly_detection/patchcore_visa.yaml
config/anomaly_detection/patchcore_visa_smoke.yaml
config/anomaly_detection/fastflow_visa.yaml
config/anomaly_detection/fastflow_visa_smoke.yaml
config/anomaly_detection/stfpm_visa.yaml
config/anomaly_detection/stfpm_visa_smoke.yaml
config/anomaly_detection/winclip_zeroshot_visa.yaml
config/anomaly_detection/winclip_zeroshot_visa_smoke.yaml
config/anomaly_detection/winclip_fewshot_visa.yaml
config/anomaly_detection/winclip_fewshot_visa_smoke.yaml
```

## 共通方針

- `PaDiM`, `PatchCore`, `FastFlow`, `STFPM` は `anomalib` ベース
- `WinCLIP` も `anomalib` ベース
- classic 系は `tuning_mode` に従って動きます
  - `none`: 単発実行
  - `grid`: grid search のみ
  - `full`: `grid search -> Optuna`
- `train.py` は学習 / チューニングだけ
- `test.py` は `experiment_summary.json` を読み、その実験の最良 run を評価する
- smoke は `visa_smoke`、本番は `visa` を使う
- smoke 用 subset は先に `src/manage_datasets/create_visa_smoke.py` で作る

## Smoke 実行前

```bash
uv run python src/manage_datasets/create_visa_smoke.py
```

## PaDiM

### 1. train 実行

```bash
uv run python src/anomaly_detection/padim/train.py \
  --config config/anomaly_detection/padim_visa.yaml
```

### 2. test 実行

```bash
uv run python src/anomaly_detection/padim/test.py \
  --config config/anomaly_detection/padim_visa.yaml
```

## PatchCore

### 1. train 実行

```bash
uv run python src/anomaly_detection/patchcore/train.py \
  --config config/anomaly_detection/patchcore_visa.yaml
```

### 2. test 実行

```bash
uv run python src/anomaly_detection/patchcore/test.py \
  --config config/anomaly_detection/patchcore_visa.yaml
```

## FastFlow

### 1. train 実行

```bash
uv run python src/anomaly_detection/fastflow/train.py \
  --config config/anomaly_detection/fastflow_visa.yaml
```

### 2. test 実行

```bash
uv run python src/anomaly_detection/fastflow/test.py \
  --config config/anomaly_detection/fastflow_visa.yaml
```

## STFPM

### 1. train 実行

```bash
uv run python src/anomaly_detection/stfpm/train.py \
  --config config/anomaly_detection/stfpm_visa.yaml
```

### 2. test 実行

```bash
uv run python src/anomaly_detection/stfpm/test.py \
  --config config/anomaly_detection/stfpm_visa.yaml
```

## WinCLIP zero-shot

### 1. test 実行

```bash
uv run python src/anomaly_detection/winclip_zeroshot/test.py \
  --config config/anomaly_detection/winclip_zeroshot_visa.yaml
```

## WinCLIP few-shot

### 1. train 実行

```bash
uv run python src/anomaly_detection/winclip_fewshot/train.py \
  --config config/anomaly_detection/winclip_fewshot_visa.yaml
```

### 2. test 実行

```bash
uv run python src/anomaly_detection/winclip_fewshot/test.py \
  --config config/anomaly_detection/winclip_fewshot_visa.yaml
```

## 評価 notebook

学習後の比較はこれを使います。

```bash
notebooks/anomaly_detection/03_evaluation.ipynb
```

この notebook では以下を比較できます。

- image-level AUROC / AP / best F1
- pixel-level AUROC / AP
- カテゴリ別精度とばらつき
- 推論結果のしきい値カーブ
- 学習データ量と精度の関係
- 実務観点の比較軸
  - 正常見逃し耐性
  - 異常取りこぼし耐性
  - 学習コスト
  - 推論コスト
  - few-shot / zero-shot 適性

## 注意

- classic anomaly models と `WinCLIP` は `anomalib` 依存です。
- 学習前に `uv sync` を実行してください。
- `visa_smoke` は smoke 専用の派生データです。本番 config は `/workspace/data/visa` を参照します。
