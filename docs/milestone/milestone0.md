# Milestone 0: ゴール定義と全体ロードマップ

## ゴール

画像処理・コンピュータビジョン案件に対して「任せてください」と言える状態を目指す。最低限、以下を満たすことを到達基準とする。

- 主要タスクごとの前処理の違いを説明できる
- 既存実装を使って学習・推論・評価・可視化まで回せる
- 精度不良の原因を、モデルではなく前処理・ラベル・分割・評価のどこにあるか切り分けできる
- 新規案件で最初に確認すべき要件を整理できる
- 1つの研究実装ではなく、複数の定番ツールチェーンを扱える

## 期間の目安

おすすめは 4 段階で 6〜10 か月。  
本業や副業と並行するなら、現実的には 8 か月前後を想定する。

- Milestone 1: 土台固め
- Milestone 2: 主要 2D タスク制覇
- Milestone 3: OCR・動画・深度・復元
- Milestone 4: 実務化

## 最低限押さえるモデル群

各タスクで 2〜3 系統を扱える状態を目安にする。

- 分類: ResNet, ConvNeXt, ViT
- 検出: YOLO11, Faster R-CNN, RTMDet
- セグメンテーション: U-Net, DeepLabv3+, SegFormer, SAM 2
- 姿勢推定: YOLO pose, HRNet または RTMPose
- OCR: PaddleOCR
- Tracking: YOLO tracking と ByteTrack 系の理解
- 深度: Depth Anything V2
- 復元: Real-ESRGAN

## 最終的に言えるようになりたいこと

このロードマップ完了時点では、少なくとも以下のような判断を言語化できる状態を目指す。

- この案件は分類ではなく検出が必要
- 精度が出ない主因はモデル容量ではなくラベルや位置ずれ
- このデータでは全面 resize より patch 化の方が安全
- validation が高すぎるので leakage を疑うべき
- OCR の前に deskew と射影補正が必要
- 深度入力を resize するなら camera intrinsics の更新が必要
