# Milestone 1: 土台固め（1〜2か月）

## 重点

モデル選定より前に、データと前処理の基礎を固める。

## 学ぶこと

- Python, PyTorch, `Dataset`, `DataLoader`
- `torchvision.transforms.v2`
- OpenCV の基本前処理: resize, crop, pad, affine, perspective transform, threshold, denoise, morphology, color space 変換
- Albumentations による image のみ、および image とラベルの同期変換
- train/val/test 分割と leakage の理解
- confusion matrix, mAP, IoU, Dice, PCK, OCR 系指標の基礎
- bbox 描画、mask 重畳、keypoint 描画、failure case gallery

## 成果物

- resize, normalize, crop, pad, visualization を含む共通前処理ライブラリ
- 解像度分布、class imbalance、破損画像、bbox/mask/keypoint 異常を確認するデータ監査ノートブック
- augmentation ごとの効き方と壊れ方を比較する検証ノート

## 到達基準

- 前処理を画像だけでなく教師ラベルまで壊さず適用できる
- ラベル種別ごとに、何を補間してよいか説明できる
- 学習前にデータ品質監査ノートを書ける
