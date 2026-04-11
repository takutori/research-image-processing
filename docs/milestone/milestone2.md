# Milestone 2: 主要 2D タスク制覇（2〜3か月）

## 重点

以下の順で主要な 2D タスクを一通り回せるようにする。

1. 画像分類
2. 画像異常検知
3. 物体検出
4. セグメンテーション
5. 姿勢推定

実務前提:
- 実案件では十分な画像とアノテーションが最初からそろっていることは少ない
- そのため supervised 学習だけでなく、zero-shot / few-shot / promptable なモデルも優先的に触る
- ただし従来の学習ベースモデルも削らずに残し、比較できるようにする

追加で触る前提のライブラリ:
- `transformers`
- `huggingface_hub`
- `datasets`
- `timm`
- `open_clip_torch`
- `sentencepiece`
- `accelerate`

API 系:
- 課金が必要な API 系モデルは、必要に応じて調査対象として名前を把握する
- ただしこの milestone ではローカル実行または公開実装があるものを優先する

## 2.1 画像分類

学ぶこと:
- resize, crop, normalize
- class imbalance 対応
- hard example 点検
- 背景リーク検査
- color augmentation

主要モデル:

| モデル | 主な位置づけ |
| --- | --- |
| ResNet | 学習あり |
| ConvNeXt | 学習あり |
| ViT | 学習あり |
| `timm` の事前学習モデル | 学習あり |
| CLIP | zero-shot / few-shot |
| SigLIP | zero-shot / few-shot |
| BLIP-2 | zero-shot / few-shot |
| Florence-2 | zero-shot / few-shot |
| DINOv2 | few-shot 寄り |
| prototypical networks | few-shot |
| linear probe / prompt tuning ベースの few-shot 分類 | few-shot |

成果物:
- 3種類の分類器比較レポート
- 背景リーク事例と対策
- 誤分類の原因分析
  - 解像度不足
  - ラベルミス
  - クラス曖昧性
  - 背景依存
- zero-shot 分類と supervised 微調整の比較
- few-shot 条件での分類ベースライン比較

## 2.2 画像異常検知

学ぶこと:
- one-class と reconstruction ベースの考え方
- 正常データ中心学習の前提
- anomaly score の設計
- patch ベース推論
- 閾値設定と false positive 点検
- 可視化で異常部位を説明する方法
- vision-language model を使った zero-shot anomaly detection の入口
- 少数正常サンプルしかないときの評価設計

主要モデル:

| モデル | 主な位置づけ |
| --- | --- |
| PaDiM | 学習あり |
| PatchCore | 学習あり |
| FastFlow | 学習あり |
| STFPM | 学習あり |
| WinCLIP | zero-shot / few-shot |
| CLIP-based anomaly detection | zero-shot / few-shot |
| AnomalyCLIP | zero-shot / few-shot |
| Rekognition Custom Labels / Azure Custom Vision | 学習あり API |
| Gemini / GPT-4o など API 系異常検知 | zero-shot API |

成果物:
- 正常/異常の判定パイプライン 1本
- anomaly map 可視化ノート
- false positive / false negative の失敗例整理
- 閾値変更時の挙動比較
- few-shot / zero-shot anomaly detection の比較メモ

## 2.3 物体検出

学ぶこと:
- bbox 追従変換
- letterbox と aspect ratio 維持
- 小物体対策
- tiling
- annotation cleaning

主要モデル:

| モデル | 主な位置づけ |
| --- | --- |
| YOLO11 | 学習あり |
| RTMDet | 学習あり |
| Faster R-CNN | 学習あり |
| RetinaNet | 学習あり |
| Grounding DINO | zero-shot / few-shot |
| OWL-ViT | zero-shot / few-shot |
| OWLv2 | zero-shot / few-shot |
| GLIP | zero-shot / few-shot |
| Florence-2 | zero-shot / few-shot |
| Detic | few-shot / open-vocabulary |
| YOLO-World | zero-shot / few-shot |
| Grounded-SAM 系パイプライン | zero-shot / few-shot |

成果物:
- COCO 形式変換ツール
- 小物体あり/なしでの前処理比較
- 失敗例分析
  - missed detection
  - false positive
  - class confusion
  - scale 問題
- open-vocabulary detection の試行メモ
- text prompt での検出パイプライン 1本

## 2.4 セグメンテーション

学ぶこと:
- image と mask の同期変換
- mask の nearest 補間
- patch 切り出し
- ignore index
- class remap

主要モデル:

| モデル | 主な位置づけ |
| --- | --- |
| U-Net | 学習あり |
| DeepLabv3+ | 学習あり |
| SegFormer | 学習あり |
| Mask R-CNN | 学習あり |
| SAM 2 | zero-shot / few-shot |
| MobileSAM | zero-shot / few-shot |
| Grounded-SAM | zero-shot / few-shot |
| SEEM | zero-shot / few-shot |
| OneFormer | 学習あり |
| CLIPSeg | zero-shot / few-shot |
| Florence-2 | zero-shot / few-shot |

成果物:
- semantic segmentation 1本
- instance segmentation 1本
- SAM 2 を使った半自動アノテーション検証
- patch と full image の比較
- promptable segmentation の実務ユースケース整理
- text / point / box prompt の違い比較

## 2.5 姿勢推定

学ぶこと:
- keypoint 追従変換
- 左右反転時のラベル交換
- visibility 管理
- person crop
- heatmap 系教師の理解
- 少数サンプルで姿勢推定を立ち上げる考え方
- detector + pose pipeline と foundation model 活用の考え方

主要モデル:

| モデル | 主な位置づけ |
| --- | --- |
| YOLO pose | 学習あり |
| HRNet | 学習あり |
| RTMPose | 学習あり |
| ViTPose | 学習あり |
| DWPose | 学習あり |
| OpenPose | 学習あり |
| Grounding DINO + pose estimator の 2 段パイプライン | few-shot 寄り |
| SAM / detector を使った person crop 補助 | few-shot 補助 |
| GPT-4o / Gemini など API 系姿勢理解 | zero-shot API |

成果物:
- flip で壊れない keypoint pipeline
- occlusion を含む評価レポート
- 単人物と複数人物の比較
- detector + pose の少数データ運用メモ
