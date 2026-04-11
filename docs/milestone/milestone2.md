# Milestone 2: 主要 2D タスク制覇（2〜3か月）

## 重点

以下の順で主要な 2D タスクを一通り回せるようにする。

1. 画像分類
2. 物体検出
3. セグメンテーション
4. 姿勢推定

## 2.1 画像分類

学ぶこと:
- resize, crop, normalize
- class imbalance 対応
- hard example 点検
- 背景リーク検査
- color augmentation

主要モデル:
- ResNet
- ConvNeXt
- ViT
- `timm` の事前学習モデル

成果物:
- 3種類の分類器比較レポート
- 背景リーク事例と対策
- 誤分類の原因分析
  - 解像度不足
  - ラベルミス
  - クラス曖昧性
  - 背景依存

## 2.2 物体検出

学ぶこと:
- bbox 追従変換
- letterbox と aspect ratio 維持
- 小物体対策
- tiling
- annotation cleaning

主要モデル:
- YOLO11
- RTMDet
- Faster R-CNN
- RetinaNet

成果物:
- COCO 形式変換ツール
- 小物体あり/なしでの前処理比較
- 失敗例分析
  - missed detection
  - false positive
  - class confusion
  - scale 問題

## 2.3 セグメンテーション

学ぶこと:
- image と mask の同期変換
- mask の nearest 補間
- patch 切り出し
- ignore index
- class remap

主要モデル:
- U-Net
- DeepLabv3+
- SegFormer
- Mask R-CNN
- SAM 2

成果物:
- semantic segmentation 1本
- instance segmentation 1本
- SAM 2 を使った半自動アノテーション検証
- patch と full image の比較

## 2.4 姿勢推定

学ぶこと:
- keypoint 追従変換
- 左右反転時のラベル交換
- visibility 管理
- person crop
- heatmap 系教師の理解

主要モデル:
- YOLO pose
- HRNet
- RTMPose

成果物:
- flip で壊れない keypoint pipeline
- occlusion を含む評価レポート
- 単人物と複数人物の比較
