# 2026-04-13 Classification Result Analysis

`notebooks/classification/03_evaluation.ipynb` で反映した結果を、artifact ベースで整理したメモ。

対象:
- `ConvNeXt V2` supervised
- `EVA-02` supervised
- `SigLIP2` supervised
- `SigLIP2` zero-shot
- `SigLIP2` few-shot
- `Florence-2` supervised
- `Florence-2` zero-shot
- `Florence-2` few-shot

## 主要結果

| Model | Mode | Best Valid Acc | Test Acc | Note |
|---|---:|---:|---:|---|
| ConvNeXt V2 | supervised | 0.9647 | 0.9264 | 強い基準線 |
| EVA-02 | supervised | 0.9647 | 0.9419 | 今回の最良 |
| SigLIP2 | supervised | 0.9538 | 0.9150 | 強いが supervised では上2つに少し負ける |
| SigLIP2 | zero-shot | 0.6889 | 0.7144 | 学習なしとしてはかなり健闘 |
| SigLIP2 | few-shot | 0.7690 | 0.7386 | zero-shot を上回るが supervised との差は大きい |
| Florence-2 | supervised | 0.0870 | 0.0690 | 今回の分類設定とは相性が悪い |
| Florence-2 | zero-shot | 0.2527 | 0.2567 | supervised よりは良いが実用水準ではない |
| Florence-2 | few-shot | 0.0476 | 0.0264 | かなり厳しい |

参照:
- `checkpoints/classification/*/experiment_summary.json`
- `checkpoints/classification/*/best_test_results/summary.json`

## supervised の比較

### 1. EVA-02 が最良

- `EVA-02` は `test_accuracy=0.9419` で今回の最良。
- `ConvNeXt V2` の `0.9264` を上回った。
- valid は両者とも `0.9647` で並んでいるので、最終的な test 一般化では `EVA-02` が少し勝った。

考察:
- `Oxford-IIIT Pet` のような中規模・細粒度寄り分類では、`EVA-02` の表現力が効いた可能性が高い。
- 一方で差は極端ではなく、`ConvNeXt V2` も十分強い。実装や運用の安定性まで含めると、今後も有力な baseline。

### 2. SigLIP2 supervised は強いが、王者ではない

- `SigLIP2` supervised は `test_accuracy=0.9150`
- zero-shot / few-shot を持つモデルとしては強いが、純 supervised 比較では `ConvNeXt V2` と `EVA-02` に届かなかった。

考察:
- `SigLIP2` は zero-shot/few-shot 側に大きな強みがあるので、単純な full fine-tune だけで評価すると本領が見えにくい。
- 実務では「少量アノテーションでどこまで行けるか」の比較に価値がある。

### 3. Florence-2 supervised は今回かなり弱い

- `Florence-2` supervised は `test_accuracy=0.0690`
- 分類器としての今回の fine-tune 実装では、明確に相性が悪い。

考察:
- `Florence-2` は promptable VLM としての使い方が中心で、今回の分類 head ベース実装では強みが出ていない可能性が高い。
- 少なくともこのタスクでは、supervised classifier としての優先度は低い。

## zero-shot / few-shot の比較

### 1. SigLIP2 zero-shot はかなり使える

- `SigLIP2 zero-shot` は `test_accuracy=0.7144`
- 学習なしでこの精度は十分に意味がある。

考察:
- 実務で「まずラベル付きデータが無い」状況の初手として有望。
- 後続の few-shot や supervised と比較する土台としても良い。

### 2. SigLIP2 few-shot は zero-shot を改善

- `SigLIP2 few-shot` は `test_accuracy=0.7386`
- zero-shot より約 `+2.4pt`

考察:
- 改善幅はあるが、まだ supervised との差は大きい。
- few-shot の改善余地:
  - `shots_per_class`
  - prompt / label text
  - head の設計
  - lr / wd
  - retrieval 的な補助

### 3. Florence-2 は zero-shot の方がまだまし

- `Florence-2 zero-shot`: `0.2567`
- `Florence-2 few-shot`: `0.0264`
- `Florence-2 supervised`: `0.0690`

考察:
- 今回の実装・設定では、`Florence-2` は classification では弱い。
- むしろ `zero-shot prompt` の方が few-shot / supervised より良い。
- このモデルは分類器として詰めるより、別タスクで使う方がよさそう。

## tuning の読み取り

### ConvNeXt V2

grid 最良:
- `valid=0.9429`
- `lr=1e-4`
- `wd=1e-4`
- `batch_size=32`
- `image_size=224`

optuna 最良:
- `valid=0.9647`
- `lr=7.30e-05`
- `wd=2.71e-04`
- `batch_size=64`
- `image_size=256`

考察:
- `Optuna` でしっかり改善している。
- `image_size=256` と `batch_size=64` が効いている。
- `ConvNeXt V2` は tuning に対して素直。

### EVA-02

grid 最良:
- `valid=0.9389`
- `lr=3e-05`
- `wd=1e-3`
- `batch_size=32`
- `image_size=224`

optuna 最良:
- `valid=0.9647`
- `lr=1.22e-05`
- `wd=5.22e-04`
- `batch_size=64`
- `image_size=224`

考察:
- `EVA-02` はかなり低い learning rate が効いている。
- `batch_size=64` まで上げた時に良くなっている。
- 画像サイズは 224 固定でも十分強い。

### SigLIP2

grid 最良:
- `valid=0.8995`
- `lr=3e-05`
- `wd=1e-4`
- `batch_size=24`
- `image_size=224`

optuna 最良:
- `valid=0.9538`
- `lr=1.17e-05`
- `wd=2.01e-05`
- `batch_size=48`
- `image_size=224`

考察:
- `SigLIP2` もかなり低い learning rate 側が効いている。
- `weight_decay` はかなり小さい方へ寄った。
- batch size は上げられるだけ上げた方が今回の設定では良かった。

### Florence-2

grid 最良:
- `valid=0.0829`
- `lr=3e-05`
- `wd=1e-4`
- `batch_size=4`
- `image_size=224`

optuna 最良:
- `valid=0.0870`
- `lr=1.30e-05`
- `wd=4.15e-05`
- `batch_size=8`
- `image_size=224`

考察:
- tuning を頑張っても改善幅は小さい。
- 今回の classification セットアップ自体が model/task fit として弱い。

## 学習曲線の読み取り

### ConvNeXt V2

- best epoch は `4`
- その後も valid は高止まり
- train はほぼ `1.0` まで上がる

考察:
- 収束は速い。
- 軽い過学習気味ではあるが、実用上はかなり安定。

### EVA-02

- best epoch は `12`
- valid が最後まで伸び切っている

考察:
- `EVA-02` はまだ学習を続ける余地がある可能性がある。
- 早期収束というより、比較的じわじわ伸びるタイプ。

### SigLIP2

- best epoch は `5`
- valid は高いが supervised 上位2つには届かない

考察:
- 収束は速い。
- 追加 epoch より、zero-shot/few-shot 方向の工夫の方がリターンが大きいかもしれない。

### Florence-2

- best epoch は `4`
- train / valid ともにほぼ伸びない

考察:
- そもそも今回の分類設定では学習が噛み合っていない。

## 結論

### 現時点の実務的な本命

1. `EVA-02 supervised`
2. `ConvNeXt V2 supervised`
3. `SigLIP2 supervised`

### アノテーション不足前提の本命

1. `SigLIP2 zero-shot`
2. `SigLIP2 few-shot`

### 今回の設定では優先度を下げてよいもの

- `Florence-2` を画像分類の中心に置く方針

## 次にやるとよいこと

1. `SigLIP2 zero-shot` の prompt 改善
2. `SigLIP2 few-shot` の tuning 強化
3. `SigLIP2` の retrieval 併用を検討
4. `EVA-02` と `ConvNeXt V2` の class-level 比較
5. `Florence-2` は分類ではなく別タスクで再評価
