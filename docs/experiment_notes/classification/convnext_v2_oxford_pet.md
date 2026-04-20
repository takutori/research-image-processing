# Classification Experiment Notes

## ConvNeXt V2 の前処理

`ConvNeXt V2` で使っている前処理は [src/classification/common/preprocessing.py](/home/taichimain/workspace/develop_research/research_image_processing/src/classification/common/preprocessing.py) にあります。設定値の基準は [config/classification/convnext_v2_oxford_pet.yaml](/home/taichimain/workspace/develop_research/research_image_processing/config/classification/convnext_v2_oxford_pet.yaml) です。

### train 時

- `RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.25))`
- `RandomHorizontalFlip(p=0.5)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)`
- `ToTensor()`
- `Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)`

### eval / test 時

- `Resize(256)`
- `CenterCrop(224)`
- `ToTensor()`
- `Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)`

### なぜこの前処理を選んでいるか

- `RandomResizedCrop`
  - Oxford-IIIT Pet は対象が比較的中央にいて十分大きいので、軽い crop augmentation が有効。
  - ただし強すぎる crop は犬猫の顔や体を切りやすいので、`scale=(0.7, 1.0)` に抑えている。
- `RandomHorizontalFlip`
  - 品種分類では左右反転でラベルは基本変わらないため、安全で効きやすい augmentation。
- `ColorJitter`
  - 室内外や照明差に少し強くする目的。
  - 品種判別では毛色や模様も重要なので、強くしすぎず light augmentation にしている。
- `Resize -> CenterCrop`
  - eval / test ではランダム性をなくして比較を安定させるため。
  - Oxford-IIIT Pet は中央寄りの構図が多く、`CenterCrop` と相性がよい。
- `Normalize(IMAGENET_MEAN, IMAGENET_STD)`
  - 使用している `convnextv2_tiny.fcmae_ft_in22k_in1k` は ImageNet 系 pretrained weight 前提なので、その入力統計に合わせている。

要するに、方針は「強すぎない augmentation で品種情報を壊さず、pretrained ConvNeXt V2 が期待する入力分布に合わせる」です。

## 実験設定

今回の実験は [config/classification/convnext_v2_oxford_pet.yaml](/home/taichimain/workspace/develop_research/research_image_processing/config/classification/convnext_v2_oxford_pet.yaml) を使い、`tuning_mode: full` で実行しています。

流れ:

1. `grid search` で粗探索
2. grid の best trial を起点に `Optuna` で微調整
3. 最良モデルを自動選択
4. その最良モデルで test を自動実行

実験結果の自動要約は [checkpoints/classification/convnext_v2/oxford_pet/experiment_summary.json](/home/taichimain/workspace/develop_research/research_image_processing/checkpoints/classification/convnext_v2/oxford_pet/experiment_summary.json) に保存されています。

## 03_evaluation.ipynb から読めること

Notebook:
- [notebooks/classification/03_evaluation.ipynb](/home/taichimain/workspace/develop_research/research_image_processing/notebooks/classification/03_evaluation.ipynb)

この notebook は、各実験について基本的に「最良モデルだけ」を基準に見ます。今回の実験では比較対象は `convnext_v2/oxford_pet` の 1 本です。

### 最終的に選ばれた最良モデル

- `best_source`: `optuna`
- `best_run_name`: `trial_023`
- `best_checkpoint_path`: `/workspace/checkpoints/classification/convnext_v2/oxford_pet/optuna/trial_023/best.pt`
- `best_valid_accuracy`: `0.9646739130434783`
- `test_accuracy`: `0.9264104660670482`

これは [experiment_summary.json](/home/taichimain/workspace/develop_research/research_image_processing/checkpoints/classification/convnext_v2/oxford_pet/experiment_summary.json) と [best_test_results/summary.json](/home/taichimain/workspace/develop_research/research_image_processing/checkpoints/classification/convnext_v2/oxford_pet/best_test_results/summary.json) の値です。

### grid search の結果

最良 trial:

- `trial_001`
- `learning_rate=0.0001`
- `weight_decay=0.0001`
- `batch_size=32`
- `image_size=224`
- `best_valid_accuracy=0.9429347826086957`

一方で `learning_rate=0.001` の trial は validation が `0.04` や `0.02` 台まで崩れており、今回の設定では明らかに高すぎました。

### Optuna の結果

最良 trial:

- `trial_023`
- `learning_rate=7.301061916628417e-05`
- `weight_decay=0.0002712810420994188`
- `batch_size=64`
- `image_size=256`
- `best_valid_accuracy=0.9646739130434783`

grid の best validation `0.9429` から、Optuna 後は `0.9647` まで改善しています。今回の tuning では、次の方向が効いています。

- learning rate を `1e-4` よりやや下げる
- image size を `224` から `256` に上げる
- batch size を `64` に上げる

### 学習曲線の読み方

最良 trial `trial_023` の history は [history.csv](/home/taichimain/workspace/develop_research/research_image_processing/checkpoints/classification/convnext_v2/oxford_pet/optuna/trial_023/history.csv) にあります。

主要な流れ:

- epoch 1: `train_acc=0.6824`, `valid_acc=0.9375`
- epoch 4: `train_acc=0.9868`, `valid_acc=0.9647` で最良
- epoch 5 以降: train はさらに上がるが、valid は少し落ちる

つまり今回の best checkpoint は、train が十分上がりつつ、validation が落ちる前の良い地点で止まっていると言えます。

### test 結果

test summary:
- `num_test_examples=3669`
- `accuracy=0.9264104660670482`

出力:
- [best_test_results/summary.json](/home/taichimain/workspace/develop_research/research_image_processing/checkpoints/classification/convnext_v2/oxford_pet/best_test_results/summary.json)
- [best_test_results/classification_report.csv](/home/taichimain/workspace/develop_research/research_image_processing/checkpoints/classification/convnext_v2/oxford_pet/best_test_results/classification_report.csv)
- [best_test_results/confusion_matrix.csv](/home/taichimain/workspace/develop_research/research_image_processing/checkpoints/classification/convnext_v2/oxford_pet/best_test_results/confusion_matrix.csv)

validation `0.9647` に対して test `0.9264` なので、極端に壊れてはいませんが、validation よりは下がっています。今後は split の安定性や seed 差、軽い regularization の見直し余地があります。

### クラス別の傾向

`classification_report.csv` から目立つ点:

- 強いクラス例
  - `Abyssinian`: `f1=0.9688`
  - `basset_hound`: `f1=0.95`
  - `beagle`: `f1=0.9340`
- 弱いクラス例
  - `american_pit_bull_terrier`: `f1=0.6977`
  - `Birman`: `f1=0.8019`
  - `Bengal`: `precision=0.6944`, `recall=1.0`, `f1=0.8197`

この傾向から、似た犬種・猫種の取り違えはまだ残っています。`03_evaluation.ipynb` の誤分類例を見ると、この種のクラス間混同を追いやすいです。

## 今回の実験からの結論

- `ConvNeXt V2` は Oxford-IIIT Pet に対してかなり強い baseline になっている。
- `grid -> Optuna` の二段階 tuning は有効で、validation は `0.9429 -> 0.9647` まで改善した。
- 最良設定は `やや小さめの learning rate`, `256 入力`, `batch size 64` 側に寄った。
- test accuracy は `0.9264` で、実装の基盤としては十分よい出発点。
- 次にやるなら:
  - seed 変更で再現性確認
  - 誤分類の定性確認
  - `OpenCLIP` や `SigLIP 2` の zero-shot / few-shot と比較
