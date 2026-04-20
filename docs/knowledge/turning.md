# Tuning Knowledge

このメモは、画像分類のハイパーパラメータチューニングについて、今回の会話で整理した内容をまとめたものです。

## まず何をチューニングするか

画像分類で最初に優先しやすいのは次です。

- `learning_rate`
- `weight_decay`
- `image_size`
- `batch_size`
- augmentation の強さ

実務や Kaggle の感覚では、特に `learning_rate` の影響が大きいです。

## `learning_rate` と `batch_size` はどちらを固定するのが普通か

結論:

- `どちらかを絶対固定するのが普通` ではない
- ただし実務では、まず `batch_size` を現実的な値に固定して、`learning_rate` を主に探すことが多い

理由:

- `learning_rate` は性能への影響が大きい
- `batch_size` は GPU メモリ制約に強く縛られる
- ただし両者は独立ではなく、相互作用がある

実務でよくある流れ:

1. まず `batch_size` を GPU に乗る現実的な値に決める
2. `learning_rate` を主に探す
3. 有望設定が見えたら `batch_size` も含めて再探索する

つまり、

- baseline 段階: `batch_size` 固定、`lr` 主探索
- 本気 tuning 段階: `batch_size` も候補に入れる

という流れが自然です。

## 今のリポジトリで採用している方針

このリポジトリでは、分類モデルの tuning は次の 2 段階にしています。

1. `grid search`
   - 粗く当たりをつける
2. `Optuna`
   - grid で良かった近傍をさらに細かく詰める

この方針は不自然ではなく、実務的にも十分妥当です。

### 重要: `image_size` は自動近傍生成しない

このリポジトリでは、`Optuna` で `image_size` を自動的に `best ± 32` のように広げるのはやめています。

理由:

- `ViT`, `EVA`, `OpenCLIP` などは入力サイズが固定のことがある
- モデル名に `..._224` のような制約が入っている場合がある
- `image_size` を勝手に広げると、モデル追加のたびに同じエラーを踏みやすい

そのため今は、

- `learning_rate`, `weight_decay` は近傍探索してよい
- `batch_size` は候補集合から選ぶ
- `image_size` は **config に明示した候補だけ使う**

という方針にしています。

## `grid search -> Optuna` の意味

これは別々の実験ではなく、1 本の tuning pipeline として考えます。

- `grid search`
  - 広い範囲を粗く見る
- `Optuna`
  - grid の最良 trial を起点に近傍を探索する

イメージ:

1. 大まかに山の場所を見つける
2. その周辺だけ細かく掘る

## Optuna とは

`Optuna` はハイパーパラメータ最適化ライブラリです。

できること:

- `learning_rate`, `weight_decay`, `batch_size` などの候補を探索する
- 過去の結果を見ながら、良さそうな値を重点的に試す
- random search より効率よく良い設定を見つけやすい

このリポジトリでは、

- `grid search` で大枠を決める
- `Optuna` でその近傍を詰める

という使い方をしています。

## Ray Tune とは

`Ray Tune` は、ハイパーパラメータ探索を大規模・並列に回すためのライブラリです。

位置づけ:

- `Optuna`
  - ハイパーパラメータ最適化ライブラリ
- `Ray Tune`
  - 最適化に加えて、分散実行や大量 trial 管理が強い基盤

強い場面:

- trial 数が多い
- 複数 GPU / 複数マシンで並列に回したい
- 大規模 tuning をしたい

今のこのリポジトリ段階では、まずは `grid -> Optuna` で十分です。

## Kaggle / 実務感で知っておくべきこと

ハイパーパラメータそのもの以外でも、次はかなり重要です。

### 1. strong baseline を先に作る

いきなり重い tuning に入るより、

- `timm` backbone
- 素直な前処理
- 最小限の augmentation

で strong baseline を 1 本作る方が重要です。

### 2. augmentation も tuning 対象

`lr` や `weight_decay` だけでなく、次もかなり効きます。

- crop の強さ
- color jitter
- mixup / cutmix
- randaugment

画像分類では augmentation の差がかなり効くことがあります。

### 3. scheduler も重要

`learning_rate` 単体ではなく、schedule 全体で見る方が実践的です。

例えば:

- cosine annealing
- warmup
- one-cycle

### 4. seed を変えて確認する

1 回だけ良かった設定は再現しないことがあります。

有望設定は:

- `seed` を変えて 2〜3 回回す

のが安全です。

### 5. validation に過適合しない

tuning 中は validation を何度も見ます。  
なので、validation だけに最適化される危険があります。

原則:

- validation は tuning に使う
- test は最後だけ使う

### 6. TTA

`TTA = Test Time Augmentation`

例えば test 時に、

- 左右反転版
- 元画像

の両方を推論して平均するようなやり方です。

Kaggle ではかなり定番ですが、本番推論速度と引き換えになります。

### 7. ensemble

最後の精度押し上げではかなり強いです。

例:

- 別モデル平均
- 別 seed 平均
- 別 fold 平均

ただし、このリポジトリではまず単体モデルをしっかり固めるのが先です。

### 8. 小さいデータでは CV を真面目に考える

小さいデータでは、1 回の split による当たり外れが大きいです。  
そのため `Cross Validation` の価値が高くなります。

## Cross Validation の意味

典型的な `K-Fold CV` は次です。

1. データを `K` 分割する
2. 1 回目は `fold1` を validation、残りを train
3. 2 回目は `fold2` を validation、残りを train
4. これを `K` 回繰り返す
5. 各 fold の validation 指標を平均する

ここで大事なのは:

- `1 fold は test` ではなく、通常は `validation`
- `test` は最後にだけ使う別データ

なので理解としては、

- `n` 個に分割して
- `n-1` 個で学習
- `1` 個で validation
- これを `n` 回回す

が正確です。

CV の目的は、

- split 依存のブレを減らす
- validation の信頼性を上げる

ことです。

## 現時点でこの repo に入っている tuning 方針

分類モデルの train.py は config を読み、次の flow を同じ入口で回します。

- `none`
  - 単発学習
- `full`
  - `grid search -> Optuna -> best model 自動選択 -> test`

これにより、

- train / test の入口をシンプルに保つ
- 同一 config から最良モデルを自動で拾う
- `03_evaluation.ipynb` で自然に比較できる

という運用にしています。

## 今後入れるとよいもの

今後さらに改善するなら優先度が高いのは次です。

1. `CV`
2. `scheduler`
3. `seed` 複数実行
4. `TTA`
5. `ensemble`

順番としては、まず `CV` と `scheduler` の方が価値が高いです。
