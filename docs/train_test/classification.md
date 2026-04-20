# Classification Train / Test

このドキュメントは、**すでに Docker コンテナに入っていて、作業ディレクトリが `/workspace` である** 前提で書いています。

## 前提

- `Oxford-IIIT Pet` をダウンロード済み
- `uv sync` 済み
- `optuna` 追加後の環境同期も済み

## 設定ファイル

train / test は、リポジトリ直下の `config/classification/*.yaml` を指定して実行します。

現在の例:

```bash
config/classification/convnext_v2_oxford_pet.yaml
config/classification/eva02_oxford_pet.yaml
config/classification/siglip2_oxford_pet.yaml
config/classification/florence2_oxford_pet.yaml
config/classification/siglip2_oxford_pet_zeroshot.yaml
config/classification/siglip2_oxford_pet_fewshot.yaml
config/classification/florence2_oxford_pet_zeroshot.yaml
config/classification/florence2_oxford_pet_fewshot.yaml
```

## ConvNeXt V2

### 1. train 実行

```bash
uv run python src/classification/convnext_v2/train.py \
  --config config/classification/convnext_v2_oxford_pet.yaml
```

この `train.py` は、設定ファイルの `tuning_mode` に従って学習・チューニングだけを行います。

- `none`: 単発学習
- `full`: `grid search -> Optuna -> best model を自動選択`

今の `convnext_v2_oxford_pet.yaml` は `full` なので、**train 実行後に最良 checkpoint が `experiment_summary.json` に記録されます。**

主な出力:

- `experiment_summary.json`
- `best_test_results/`
- `grid/`
- `optuna/`

### 2. test だけ再実行

```bash
uv run python src/classification/convnext_v2/test.py \
  --config config/classification/convnext_v2_oxford_pet.yaml
```

この `test.py` は、同じ設定ファイルから対応する `experiment_summary.json` を読み、**その実験で最良と判定された checkpoint** を自動で使います。`best_test_results/` を生成し、`experiment_summary.json` に `test_accuracy` を追記します。

## 評価 notebook

学習後の比較や誤分類分析はこれを使います。

```bash
notebooks/classification/03_evaluation.ipynb
```

この notebook では以下を比較できます。

- 最良モデル同士の validation / test の主要指標
- 最良モデルの学習曲線
- クラス別 precision / recall / f1-score
- confidence 分布
- 誤分類例
- tuning 中の指標推移

## EVA-02

### 1. train 実行

```bash
uv run python src/classification/eva02/train.py \
  --config config/classification/eva02_oxford_pet.yaml
```

`EVA-02` も `ConvNeXt V2` と同じで、設定ファイルの `tuning_mode` に従って動きます。

- `none`: 単発学習
- `full`: `grid search -> Optuna -> best model を自動選択`

今の `eva02_oxford_pet.yaml` も `full` 前提なので、**train 実行後に最良 checkpoint が `experiment_summary.json` に記録されます。**

主な出力:

- `experiment_summary.json`
- `best_test_results/`
- `grid/`
- `optuna/`

### 2. test だけ再実行

```bash
uv run python src/classification/eva02/test.py \
  --config config/classification/eva02_oxford_pet.yaml
```

この `test.py` も、対応する `experiment_summary.json` を読み、**その実験で最良と判定された checkpoint** を自動で使います。

## 評価 notebook の使い方

`03_evaluation.ipynb` は `checkpoints/classification/*/*/experiment_summary.json` を読むので、`ConvNeXt V2` と `EVA-02` の両方を学習したあとに再実行すると、そのまま最良モデル同士を比較できます。

## SigLIP 2

### 1. train 実行

```bash
uv run python src/classification/siglip2/train.py \
  --config config/classification/siglip2_oxford_pet.yaml
```

`SigLIP 2` も `ConvNeXt V2` / `EVA-02` と同じで、設定ファイルの `tuning_mode` に従って動きます。

- `none`: 単発学習
- `full`: `grid search -> Optuna -> best model を自動選択`

今の `siglip2_oxford_pet.yaml` は `full` 前提なので、**train 実行後に最良 checkpoint が `experiment_summary.json` に記録されます。**

主な出力:

- `experiment_summary.json`
- `best_test_results/`
- `grid/`
- `optuna/`

### 2. test だけ再実行

```bash
uv run python src/classification/siglip2/test.py \
  --config config/classification/siglip2_oxford_pet.yaml
```

この `test.py` も、対応する `experiment_summary.json` を読み、**その実験で最良と判定された checkpoint** を自動で使います。

## Florence-2

### 1. train 実行

```bash
uv run python src/classification/florence2/train.py \
  --config config/classification/florence2_oxford_pet.yaml
```

`Florence-2` も同様に、設定ファイルの `tuning_mode` に従って動きます。

- `none`: 単発学習
- `full`: `grid search -> Optuna -> best model を自動選択`

今の `florence2_oxford_pet.yaml` は `full` 前提なので、**train 実行後に最良 checkpoint が `experiment_summary.json` に記録されます。**

主な出力:

- `experiment_summary.json`
- `best_test_results/`
- `grid/`
- `optuna/`

### 2. test だけ再実行

```bash
uv run python src/classification/florence2/test.py \
  --config config/classification/florence2_oxford_pet.yaml
```

この `test.py` も、対応する `experiment_summary.json` を読み、**その実験で最良と判定された checkpoint** を自動で使います。

## SigLIP 2 zero-shot / few-shot

### zero-shot

```bash
uv run python src/classification/siglip2/zeroshot.py \
  --config config/classification/siglip2_oxford_pet_zeroshot.yaml
```

出力形式は supervised 実験と揃えてあり、`experiment_summary.json` と `best_test_results/` を生成します。
そのため `03_evaluation.ipynb` で他モデルと一緒に比較できます。

### few-shot

```bash
uv run python src/classification/siglip2/fewshot.py \
  --config config/classification/siglip2_oxford_pet_fewshot.yaml
```

few-shot は `shots_per_class` を使って train split から少量サンプルを抽出し、分類 head を学習します。

## Florence-2 zero-shot / few-shot

### zero-shot

```bash
uv run python src/classification/florence2/zeroshot.py \
  --config config/classification/florence2_oxford_pet_zeroshot.yaml
```

Florence-2 は prompt ベースで breed 名を推定し、予測文字列を候補ラベルへ正規化して評価します。

### few-shot

```bash
uv run python src/classification/florence2/fewshot.py \
  --config config/classification/florence2_oxford_pet_fewshot.yaml
```

few-shot は `shots_per_class` を使って train split から少量サンプルを抽出し、分類 head を学習します。

## 注意

- `SigLIP 2` と `Florence-2` は `transformers` 系依存が必要です。
- まだ環境同期していない場合は、学習前に `uv sync` を実行してください。
