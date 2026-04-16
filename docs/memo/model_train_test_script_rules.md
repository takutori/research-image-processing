# Model Train/Test Script Rules

このメモは、モデルごとの `train.py`, `test.py`, config, 評価 artifact を追加・更新するときのルールです。

## 読むタイミング

- モデルの学習 / テスト用スクリプトを作るとき

## 基本方針

- `train.py` は学習だけを担当する
- `test.py` は評価だけを担当する
- `train.py` が自動で `test` まで実行してはいけない
- 実験設定は CLI に大量列挙せず、`config/` 配下の YAML を読む

## ディレクトリ構成

- `src/<task>/<model_name>/`
  - `config.py`
  - `model.py`
  - `train.py`
  - `test.py`
- 共通処理は `src/<task>/common/`
- config は `config/<task>/`
- notebook は `notebooks/<task>/`
- checkpoint は `checkpoints/<task>/<model_name>/`

### モデル名フォルダの切り方

- 同じ数理モデルでも、実験の性質が変わるなら別フォルダとして切る
- 例えば次は別モデル扱いにする
  - supervised
  - zeroshot
  - fewshot
  - 使う backbone / checkpoint / processor が変わる場合
  - 入力画像サイズの違いで呼び出す model variant 自体が変わる場合
- つまり `config/<task>/` の粒度に `src/<task>/` も合わせる
- 「同じ理論モデルだから 1 フォルダにまとめる」より、「実装・依存・入出力・評価方式が同じ単位」で切る

## config ルール

- 本番用 config と smoke/quick 用 config を分ける
- 本番用は `<model>_<dataset>.yaml`
- 軽量確認用は `<model>_<dataset>_smoke.yaml` や `<model>_<dataset>_quick.yaml`
- 同じモデルでも
  - supervised
  - zeroshot
  - fewshot
  は別 config に分ける

## tuning ルール

- tuning は `grid search -> Optuna` の順で行う
- `grid` は粗探索
- `Optuna` は `grid` 最良近傍の詳細探索
- ただし探索範囲は model 制約を無視してはいけない

### 特に注意すること

- `image_size` を勝手に近傍生成しない
- `image_size` は config に書かれた候補だけ使う
- モデル名に固定入力サイズが含まれる場合は、その制約を事前にチェックする
  - 例: `..._224` なのに `256` を試さない
- `batch_size` はメモリ依存なので、保守的に始める

## 出力 artifact ルール

`notebooks/<task>/03_evaluation.ipynb` で比較できるよう、出力形式を揃える。

最低限必要なもの:
- `experiment_summary.json`
- `best_test_results/summary.json`
- `best_test_results/predictions.csv`
- `best_test_results/classification_report.csv`

学習がある場合:
- `history.csv`

tuning がある場合:
- `grid/tuning_results.csv`
- `optuna/optuna_results.csv`

## experiment_summary.json ルール

最低限入れるもの:
- `experiment_name`
- `mode`
- `best_source`
- `best_run_name`
- `best_checkpoint_path`
- `best_valid_accuracy`

`test.py` 実行後は:
- `test_accuracy`
も追記する

## test.py ルール

- 同じ config を読んで、その実験の `experiment_summary.json` から最良 checkpoint を拾う
- ユーザーが毎回 checkpoint path を手で指定しなくてよい形にする
- `test.py` 実行後に `experiment_summary.json` を更新して `test_accuracy` を書く

## smoke / quick ルール

- smoke/quick は「最後まで流れるか」を見るためのもの
- 精度評価を主目的にしない
- 本番 config と artifact 形式は揃える
- notebook 側で混ぜて読んでも壊れないようにする

## notebook との整合

- `notebooks/<task>/03_evaluation.ipynb` で読める形式に揃える
- 比較軸が異なるものは無理に同列比較しない
  - 学習なし zero-shot に学習曲線はない
  - `history.csv` が無いものは学習曲線比較から除外する

## Codex 実行ルール

- Codex は本番学習 / 本番テストを実行しない
- smoke であっても、学習 / テストの実行は人間が行う

理由:
- 学習実行中のターミナル出力は非常に長くなりやすい
- Codex がその出力を読むと、すぐにコンテキスト制限に達しやすい
- その結果、実装・修正・分析の作業継続性が落ちる

## Codex がやってよい確認

- `python3 -m compileall`
- 設定値やパスの静的確認
- notebook / docs / config の整合確認

## 追加で意識すること

- モデルごとの制約は `model.py` または `train.py` 側で早めにチェックする
- ランタイム依存のモデルは、ライセンスと公開可否を先に確認する
- zero-shot / few-shot も supervised と同じ artifact 形式にできるなら揃える
- **モデルごとに実務に沿った最適な実装を目指す**
  - 既存モデルの設計を「踏襲する」ことを原則にしない
  - 学習方式・使用ライブラリ・データ要件がモデルによって異なる場合は、それに合った実装を選ぶ
  - 例: 高レベルライブラリ（anomalib 等）を使うモデルは、そのライブラリの流儀に従う。自前 PyTorch ループを無理に合わせない
  - 例: epoch のないモデル（PaDiM 等）に epoch 前提の仕組み（Optuna pruning 等）を無理に適用しない
  - artifact の出力形式（CSV・JSON の構造）は揃えるが、実装の内部構造は揃えなくてよい
  - 「既存と同じ構造にするか」より「このモデルの性質に合った最短の実装か」を優先する
- **自前実装はなるべく避け、使用できるライブラリを積極的に使う**
  - 標準ライブラリ・OSS で実現できる処理を自前で書かない
  - ライブラリの API・流儀に乗ることで、実装量・バグ・メンテコストを減らす
  - 自前実装が必要になるのは「ライブラリで対応できない要件が明確にある場合」のみ
  - ライブラリの使い方が不明な場合は、ソースや公式ドキュメントを先に調べてから実装を始める
