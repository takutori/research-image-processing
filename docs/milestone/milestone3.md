# Milestone 3: OCR・動画・深度・復元（2〜3か月）

## 3.1 OCR / ドキュメント AI

学ぶこと:
- deskew
- perspective correction
- threshold
- morphology
- layout 切り出し
- 文字領域抽出

主要モデル:
- PaddleOCR
- DB 系 text detection
- CRNN または SVTR 系 text recognition
- PaddleOCR 系の文書解析モデル

成果物:
- 斜め画像 OCR パイプライン
- 帳票・スクショ・自然画像文字の比較
- 認識失敗の原因分解
  - 傾き
  - 低解像度
  - 反射
  - レイアウト崩れ

## 3.2 Tracking / 動画理解

学ぶこと:
- fps 正規化
- clip 化
- フレーム欠損対応
- camera motion 補正
- detection と tracking の接続

主要モデル:
- YOLO tracking
- ByteTrack
- BoT-SORT
- TimeSformer と VideoMAE の概念理解

成果物:
- MOT パイプライン 1 本
- ID switch の要因分析
- fps 差が性能に与える影響の検証

## 3.3 深度推定 / 3D の入口

学ぶこと:
- intrinsics / extrinsics
- resize 時の camera parameter 更新
- undistort
- stereo / multiview の基礎
- point cloud の軽い前処理

主要モデル:
- Depth Anything V2
- Prompt Depth Anything
- Video Depth Anything

成果物:
- monocular depth パイプライン 1 本
- resize 時に intrinsics 更新あり/なしの比較
- 動画深度の temporal consistency 確認

## 3.4 画像復元

学ぶこと:
- synthetic degradation 作成
- paired / unpaired データ整備
- noise / blur / compression artifact の再現
- patch 化

主要モデル:
- Real-ESRGAN
- DnCNN や超解像系の基礎理解
- diffusion 系復元は概要理解まで

成果物:
- blur / noise / jpeg 劣化を人工生成して学習する構成
- 実画像で劣化モデルがずれる例の整理
- PSNR / SSIM と見た目評価のズレのまとめ
