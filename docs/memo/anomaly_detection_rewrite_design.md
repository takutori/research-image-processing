# 異常検知モジュール 再設計方針

## 背景・経緯

既存の `src/anomaly_detection/` は分類タスクの設計を無理に踏襲したことで以下の問題が生じた。

- anomalib の Folder datamodule に合わせるため per-trial でシンボリックリンク staging を生成・破棄しており、実行時間が異常に長かった（PaDiM 全カテゴリで 15 時間以上）
- epoch のない PaDiM / PatchCore に Optuna pruning を設計に含めており、チューニングが無意味に複雑だった
- `evaluate_run()` が fit + predict を一体化していたため、test.py が学習を再実行するバグの温床になっていた

これらの反省を踏まえ `src/anomaly_detection/` を削除し、ゼロから書き直す。

## データ読み込み

- anomalib の `Visa` datamodule（`anomalib.data.Visa`）を使う
- データ構造は `{data_root}/visa/visa_pytorch/` として anomalib が期待する形に整える
- `visa_pytorch/` はシンボリックリンクで一度だけ構築し、以降全実験で使い回す
- 構築処理は `train.py` 冒頭で自動実行（`visa_pytorch/` が既存なら何もしない）
- シンボリックリンク構築の元は `{data_root}/visa/split_csv/1cls.csv`（既存）

## データ分割

```
1cls.csv train（正常のみ）
  ├── 80% → 実際の学習に使う
  └── 20% → hold-out（SWSA で合成異常を生成 → validation セット）

1cls.csv test（正常 + 異常）→ 最終 test のみ使う
```

- anomaly ラベルは最終 test のみに使う
- `val_split_mode="from_test"` は使わない

## カテゴリ指定

- config で `categories: [candle]` のように指定する
- 省略時は全 12 カテゴリ（`list_visa_categories()` で列挙）
- **本番実験は candle 単体で回す方針**（他カテゴリへの展開は後回し）

## チューニング方針

### PaDiM / PatchCore

- **grid のみ**（Optuna なし）
- 理由: 探索空間が小さく全カテゴリ変数なので grid で十分
- pruning なし（epoch がないため）
- validation 指標: 合成異常 AUROC（SWSA + DTD テクスチャ）

探索するハイパーパラメータ:

| モデル | パラメータ | 候補例 |
|--------|-----------|--------|
| PaDiM | backbone, image_size, n_features | resnet18/wide_resnet50_2, 224/320, 100/200 |
| PatchCore | backbone, image_size, coreset_sampling_ratio | resnet18/wide_resnet50_2, 224/320, 0.1/0.01 |

### FastFlow / STFPM

- **grid → Optuna**（pruning あり）
- validation 指標: training loss（epoch ごとに `trial.report()` / `trial.should_prune()`）
- pruner: `optuna.pruners.MedianPruner`

## 合成異常生成（SWSA）

- anomalib の `PerlinAnomalyGenerator` + DTD テクスチャを使う
- DTD は `src/manage_datasets/download_dtd.py` でダウンロード（保存先: `{data_root}/dtd/`）
- 正常画像 hold-out の 20% に適用し、合成異常 + マスクを生成する
- PaDiM / PatchCore の grid tuning validation のみで使う

## ファイル構成

```
src/anomaly_detection/
  common/
    __init__.py
    data.py          # visa_pytorch/ セットアップ, カテゴリ列挙
    evaluation.py    # AUROC, AP, F1, pixel metrics
    artifacts.py     # JSON / CSV 保存
    synthetic.py     # SWSA: hold-out 分割 + 合成異常生成ラッパー
  padim/
    config.py
    model.py
    train.py         # grid tuning → best checkpoint 保存
    test.py          # best checkpoint で predict のみ
  patchcore/
    config.py
    model.py
    train.py
    test.py
  fastflow/
    config.py
    model.py
    train.py         # grid → optuna + pruning → best checkpoint 保存
    test.py
  stfpm/
    config.py
    model.py
    train.py
    test.py
  winclip_zeroshot/
    config.py
    model.py
    test.py          # train.py なし（zero-shot）
  winclip_fewshot/
    config.py
    model.py
    train.py         # reference images の登録のみ
    test.py
```

## artifact 出力形式

```
checkpoints/anomaly_detection/<model>/<experiment>/
  experiment_summary.json       # best_artifact_path, best_image_level_metric 等
  grid/
    tuning_results.csv          # 全 grid trial の valid 精度
  optuna/                       # FastFlow / STFPM のみ
    optuna_results.csv
    optuna_summary.json
  best_test_results/
    predictions.csv
    category_metrics.csv
    image_level_metrics.json
    pixel_level_metrics.json
    threshold_curve.csv
    summary.json
```

## experiment_summary.json 必須フィールド

```json
{
  "experiment_name": "...",
  "mode": "grid" | "full" | "none",
  "best_source": "grid" | "optuna" | "train",
  "best_artifact_path": "...",
  "best_image_level_metric": 0.0,
  "best_pixel_level_metric": 0.0,
  "test_image_level_metric": null,
  "test_pixel_level_metric": null
}
```

`test.py` 実行後に `test_image_level_metric` / `test_pixel_level_metric` を追記する。

## その他

- `num_workers: 12`（物理 16 コアのマシン）
- config は全モデル作り直し
- `train.py` は学習のみ、`test.py` は predict のみ（自動連鎖禁止）
- zero-shot（WinCLIP zeroshot）は `test.py` のみ
- `python3 -m compileall` による静的確認のみ行う（実行はユーザーが行う）
