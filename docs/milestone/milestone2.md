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
- このリポジトリでは、実装の主対象は基本的に `frontier` のモデルとする
- `current` は必要な比較ベースラインとして最小限だけ触る
- `legacy` は理論理解用とし、原則として実装優先度は下げる

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

| カテゴリ | 世代感 | モデル | 今の見立て |
| --- | --- | --- | --- |
| 学習あり | legacy | ResNet | 古典的な基準モデル |
| 学習あり | current | ConvNeXt | 実務でまだ強い定番 |
| 学習あり | frontier | ConvNeXt V2 | supervised 側の有力本命 |
| 学習あり | current | ViT | Transformer 分類の基礎 |
| 学習あり | frontier | EVA-02 | 学習あり側の最前線候補 |
| 学習あり | current | `timm` の事前学習モデル | 実装実務の標準入口 |
| zero-shot / few-shot | current | CLIP | zero-shot の基礎標準 |
| zero-shot / few-shot | current | OpenCLIP | open 実装の実務本命 |
| zero-shot / few-shot | current | SigLIP | CLIP 系の強い系譜 |
| zero-shot / few-shot | frontier | SigLIP 2 | zero-shot 側の最前線候補 |
| zero-shot / few-shot | current | BLIP-2 | 分類専用ではないが強力 |
| zero-shot / few-shot | frontier | Florence-2 | 多用途 VLM として有力 |
| few-shot 寄り | current | DINOv2 | 少量ラベルで強い特徴抽出器 |
| few-shot | legacy | prototypical networks | few-shot の古典手法 |
| few-shot 寄り | current | linear probe | 強い backbone の基本比較軸 |
| few-shot | current | prompt tuning | VLM 適応の基本手段 |
| few-shot | current | CoOp | CLIP few-shot の定番 |
| few-shot | current | CoCoOp | unseen class に強い発展形 |
| few-shot | frontier | Tip-Adapter | training-free 寄りの本命 |
| few-shot | frontier | Tip-Adapter-F | Tip-Adapter の強化版 |

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

| カテゴリ | 世代感 | モデル | 学習あり / zero-shot / few-shot | 今の見立て |
| --- | --- | --- | --- | --- |
| classic backbone-free stats | current | PaDiM | 学習あり | まず置く基準線。正常分布ベースで堅実 |
| classic memory bank | current | PatchCore | 学習あり | 実務でも強い定番。まず試す価値が高い |
| classic flow-based | current | FastFlow | 学習あり | anomaly map 比較で押さえたい有力候補 |
| classic teacher-student | current | STFPM | 学習あり | 局在比較向き。teacher-student 系の代表 |
| efficient supervised | frontier | EfficientAD | 学習あり | 高速性まで含めて実務で重要な frontier 候補 |
| zero/few-shot VLM | current | WinCLIP / WinCLIP+ | zero-shot / few-shot | 現実的な zero/few-shot 本命 |
| zero-shot prompt-adapted VLM | frontier | AdaCLIP | zero-shot | frontier 寄りの有力候補。WinCLIP の次に見る価値が高い |
| generic CLIP baseline | current | CLIP-based anomaly detection | zero-shot / few-shot | 比較用ベースラインとして有用 |
| managed API | current | Rekognition Custom Labels / Azure Custom Vision | 学習あり API | 実務 PoC 向け。学習は速いが API 前提 |
| foundation-model API | frontier | Gemini / GPT-4o など API 系異常検知 | zero-shot API | 調査価値はあるが repo 実装の主対象ではない |

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

| カテゴリ | 世代感 | モデル | 学習あり / zero-shot / few-shot | 今の見立て |
| --- | --- | --- | --- | --- |
| two-stage anchor-based | legacy | Faster R-CNN | 学習あり | two-stage の教科書モデル。精度は高いが遅い。理論理解に有用 |
| one-stage anchor-based | legacy | RetinaNet | 学習あり | focal loss 提案元。one-stage と class imbalance の基礎理解に使う |
| one-stage anchor-free | current | YOLOv8 | 学習あり | ultralytics エコシステムの現実務標準。まず置く基準線 |
| one-stage anchor-free | frontier | YOLO11 | 学習あり | ultralytics 最新版。YOLOv8 後継として supervised 側の本命 |
| one-stage anchor-free | current | RTMDet | 学習あり | mmdetection 系。軽量高速で実務向き。YOLO との比較に有用 |
| Transformer-based | current | DETR | 学習あり | end-to-end Transformer 検出の起点。anchor 不要の基礎理解に使う |
| Transformer-based | frontier | DINO (IDEA-Research) | 学習あり | DETR 系の精度最前線。deformable attention で小物体にも強い |
| Transformer-based | frontier | RT-DETR v2 | 学習あり | リアルタイム DETR。速度と精度の両立で YOLO 系と比較価値が高い |
| open-vocabulary / zero-shot | current | GLIP | zero-shot / few-shot | grounding 系の先駆け。Grounding DINO の前世代として理論把握に有用 |
| open-vocabulary / zero-shot | current | OWL-ViT | zero-shot / few-shot | Google 製 open-vocabulary 検出の基礎モデル |
| open-vocabulary / zero-shot | frontier | OWLv2 | zero-shot / few-shot | OWL-ViT の強化版。精度大幅向上で zero-shot 側の有力候補 |
| open-vocabulary / zero-shot | frontier | Grounding DINO | zero-shot / few-shot | テキストプロンプト検出の現実務本命。zero-shot 側でまず試す |
| open-vocabulary / zero-shot | frontier | YOLO-World | zero-shot / few-shot | YOLO 系の速度で open-vocabulary を実現。実務バランス候補 |
| open-vocabulary / few-shot | current | Detic | few-shot / open-vocabulary | image-level ラベルで語彙を大幅拡張。open-vocabulary 寄りの手法 |
| multi-modal VLM | frontier | Florence-2 | zero-shot / few-shot | 多用途 VLM として検出も可。プロンプト次第で柔軟に使える |
| composite pipeline | frontier | Grounded-SAM 系パイプライン | zero-shot / few-shot | Grounding DINO + SAM の連結。検出→分割まで一貫して扱える |

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
- semantic / instance / panoptic の違いと使い分け
- promptable segmentation の実務ユースケース
- zero-shot セグメンテーションとアノテーションコスト削減

主要モデル:

| カテゴリ | 世代感 | モデル | 学習あり / zero-shot / few-shot | 今の見立て |
| --- | --- | --- | --- | --- |
| semantic encoder-decoder CNN | legacy | U-Net | 学習あり | semantic / 医療系の教科書モデル。encoder-decoder 構造の基礎理解に使う |
| semantic encoder-decoder CNN | current | DeepLabv3+ | 学習あり | atrous conv と ASPP の理解に有用。CNN 系 semantic seg の基準線 |
| semantic encoder-decoder Transformer | current | SegFormer | 学習あり | 軽量 Transformer。Mix-FFN と階層エンコーダで速度と精度を両立。実務向き |
| instance segmentation | current | Mask R-CNN | 学習あり | two-stage instance seg の教科書。Faster R-CNN + mask head の構造理解に有用 |
| instance segmentation | current | YOLOv8-seg / YOLO11-seg | 学習あり | ultralytics エコシステムで instance seg も one-stage で扱える。実務の現標準候補 |
| instance segmentation | current | RTMDet-Ins | 学習あり | mmdet 系の軽量 instance seg。YOLO 系との速度・精度比較に有用 |
| unified (semantic / instance / panoptic) | frontier | Mask2Former | 学習あり | masked attention で全セグタスクを統一。精度最前線。supervised 側の本命 |
| unified (semantic / instance / panoptic) | frontier | OneFormer | 学習あり | テキスト条件付きクエリで 3 タスクを 1 モデルで扱う。Mask2Former の次に見る価値が高い |
| promptable zero-shot | current | SAM | zero-shot / few-shot | promptable segmentation の起点。point / box / mask prompt で汎用的に使える |
| promptable zero-shot | frontier | SAM 2 | zero-shot / few-shot | 動画対応・精度向上版。静止画でも SAM より実務本命 |
| promptable zero-shot 軽量 | current | MobileSAM | zero-shot / few-shot | SAM のエッジ向け蒸留版。速度優先ユースケースに有用 |
| open-vocabulary zero-shot | current | CLIPSeg | zero-shot / few-shot | CLIP 特徴をデコードしてテキスト → mask。text-prompted seg の基礎理解に使う |
| open-vocabulary zero-shot | frontier | SEEM | zero-shot / few-shot | point / text / box / 参照画像など多様なプロンプトを統一。promptable seg の frontier 候補 |
| composite pipeline | frontier | Grounded-SAM 系パイプライン | zero-shot / few-shot | Grounding DINO + SAM の連結。テキスト → 検出 → mask まで一貫して扱える実務本命 |
| multi-modal VLM | frontier | Florence-2 | zero-shot / few-shot | 検出・分類・seg を統一 VLM で扱う。プロンプト次第でセグ出力も可 |

セグメンテーションの種類:

**Semantic Segmentation（意味的分割）**
- ピクセルごとにクラスラベルを付ける
- 同じクラスの物体は個体を区別しない（犬が2匹いても同じ「犬」ラベル）
- 出力は画像と同サイズの class index マップ（W × H × 1）
- 典型的な損失関数: cross entropy loss、dice loss
- 評価指標: mIoU（mean Intersection over Union）
- 実務ユースケース: 路面・空・建物などの領域分類、農地判別、医療画像の臓器分割

**Instance Segmentation（インスタンス分割）**
- 同じクラスでも個体ごとに別マスクとして出力する
- 物体検出の延長：bbox に加えて各インスタンスのピクセルマスクを出力
- 出力は各インスタンスのバイナリマスク（個数 × W × H）+ クラスラベル + スコア
- 背景ピクセルはどのインスタンスにも属さない扱い（stuff 領域は対象外）
- 評価指標: mask AP（COCO 形式、IoU 閾値ごとの平均）
- 実務ユースケース: 個体カウント（農業・工場検査）、物体追跡、アノテーション補助

**Panoptic Segmentation（パノプティック分割）**
- Semantic + Instance を統合した最も情報量の多いタスク
- 画像内の全ピクセルをカバーする（Semantic は背景も分類、Instance は背景を無視）
- things（数えられる物体: 人・車・犬）はインスタンス単位で識別
- stuff（数えられない背景: 空・道路・芝生）はクラス単位で分類
- 出力は各ピクセルに対して (class_id, instance_id) のペアを持つパノプティックマップ
- 評価指標: PQ（Panoptic Quality）= SQ（Segmentation Quality）× RQ（Recognition Quality）
- 実務ユースケース: 自動運転のシーン理解、ロボットナビゲーション

**3タスクの対比まとめ:**

| 観点 | Semantic | Instance | Panoptic |
| --- | --- | --- | --- |
| 個体識別 | なし | あり | あり（things のみ） |
| 背景カバー | あり | なし | あり |
| 全ピクセルを分類 | ✓ | ✗（背景は無視） | ✓ |
| 出力形式 | class マップ | マスク × N 個 | (class, instance) ペアマップ |
| アノテーションコスト | 中 | 高 | 最高 |
| 代表モデル | SegFormer, DeepLabv3+ | Mask R-CNN, YOLO-seg | Mask2Former, OneFormer |

成果物:
- semantic segmentation 1本（Oxford-IIIT Pet trimaps を使った U-Net または SegFormer）
- instance segmentation 1本（COCO instances を使った Mask R-CNN または YOLO11-seg）
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
