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

| カテゴリ | 世代感 | モデル | 学習あり / zero-shot / few-shot | 今の見立て |
| --- | --- | --- | --- | --- |
| OCR pipeline | current | PaddleOCR / PP-OCRv5 pipeline | 学習あり / zero-shot | 実務の標準入口。text detection + recognition を一体でまず試す本命 |
| text detection | current | DB / DBNet 系 | 学習あり / zero-shot | OCR detection の基本線。文字領域抽出の土台として理解必須 |
| text detection | frontier | PP-OCRv5 det | 学習あり / zero-shot | PaddleOCR 系の現行主力。手書き・歪み・縦書きも含めて改善が大きい |
| text recognition | legacy | CRNN | 学習あり / zero-shot | OCR recognition の古典。CTC 系 recognition の理解に有用 |
| text recognition | current | SVTR | 学習あり / zero-shot | scene text recognition の有力候補。CRNN より modern な比較軸 |
| text recognition | frontier | PP-OCRv5 rec | 学習あり / zero-shot | PaddleOCR 系の現行主力。多言語・縦書き・難文字条件まで広く強い |
| document parsing pipeline | current | PP-StructureV3 | 学習あり / zero-shot | layout / table / formula / reading order を含む実務本命パイプライン |
| document VLM | frontier | PaddleOCR-VL | zero-shot / few-shot | 文書解析 VLM の有力候補。複雑レイアウトや表・数式まで含めて強い |

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

| カテゴリ | 世代感 | モデル | 学習あり / zero-shot / few-shot | 今の見立て |
| --- | --- | --- | --- | --- |
| tracking-by-detection pipeline | current | Ultralytics YOLO Track | 学習あり / zero-shot | まず動かす入口。detect と tracking を短距離で接続しやすい |
| MOT tracker | current | ByteTrack | 学習あり / zero-shot | tracking-by-detection の基準線。本質は低 score detection も捨てずに結び直す発想 |
| MOT tracker + ReID / CMC | current | BoT-SORT | 学習あり / zero-shot | ByteTrack の次に見る実務候補。ReID と camera motion compensation まで含めて強い |
| video transformer | current | TimeSformer | 学習あり / zero-shot | video understanding に Transformer 軸を作った代表。概念理解に有用 |
| self-supervised video backbone | current | VideoMAE | 学習あり / zero-shot / few-shot | 動画表現学習の有力基盤。少量ラベル適応や downstream で重要 |
| self-supervised video backbone | frontier | VideoMAE V2 | 学習あり / zero-shot / few-shot | VideoMAE の強化版。実務や比較軸としては V1 よりこちらを優先したい |

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

| カテゴリ | 世代感 | モデル | 学習あり / zero-shot / few-shot | 今の見立て |
| --- | --- | --- | --- | --- |
| monocular foundation depth | frontier | Depth Anything V2 | zero-shot / few-shot | monocular depth の本命。まず試すべき foundation model |
| prompted metric depth | frontier | Prompt Depth Anything | zero-shot / few-shot | low-res LiDAR prompt を使って metric depth を上げる発展形。ロボティクス寄りで重要 |
| video monocular depth | frontier | Video Depth Anything | zero-shot / few-shot | 動画深度の temporal consistency を見る主力候補。長尺動画まで意識されている |
| monocular metric depth | frontier | Depth Pro | zero-shot | zero-shot metric depth の有力候補。高解像度で sharp な深度を高速に出せる |
| universal metric depth | frontier | UniDepth / UniDepthV2 | zero-shot / few-shot | camera 条件差を跨いで使いやすい metric depth 候補。Depth Anything 系との比較価値が高い |

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

| カテゴリ | 世代感 | モデル | 学習あり / zero-shot / few-shot | 今の見立て |
| --- | --- | --- | --- | --- |
| denoising CNN | legacy | DnCNN | 学習あり / zero-shot | 画像復元の古典。noise residual 学習の基礎理解に有用 |
| blind real-world super-resolution | current | Real-ESRGAN | 学習あり / zero-shot | 実画像寄り超解像の入口として最重要。まず試す候補 |
| transformer restoration | current | SwinIR | 学習あり / zero-shot | super-resolution / denoise / deblock を広く扱える代表。復元系の強い比較軸 |
| efficient image restoration | frontier | NAFNet | 学習あり / zero-shot | denoise / deblur の modern baseline。本番実装寄りで価値が高い |
| blind image restoration diffusion pipeline | frontier | DiffBIR | zero-shot / few-shot | blind restoration を diffusion prior で扱う有力候補。概要理解枠として最有力 |
| general restoration pipeline | current | BasicSR 系エコシステム | 学習あり / zero-shot | 個別モデルよりも実装基盤として重要。Real-ESRGAN や NAFNet の土台として押さえたい |

成果物:
- blur / noise / jpeg 劣化を人工生成して学習する構成
- 実画像で劣化モデルがずれる例の整理
- PSNR / SSIM と見た目評価のズレのまとめ
