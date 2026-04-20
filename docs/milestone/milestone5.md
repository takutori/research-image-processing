# Milestone 5: GPU 最適化と高速化（1〜2か月）

## 重点

「動く」状態から、「実務で十分な速度・コスト・安定性で回せる」状態へ移る。  
Milestone 4 の ONNX, TensorRT, 軽量化の入口を、GPU 実装と推論高速化の観点で実戦レベルまで引き上げる。

## 学ぶこと

- CUDA Toolkit の役割理解
  - Driver と Toolkit の違い
  - `nvcc` が必要になるケース
  - PyTorch wheel に CUDA runtime が同梱されるケース
- GPU 実行環境の整理
  - ローカル Docker Desktop GPU 連携
  - AWS 上の GPU 実行基盤
  - SageMaker と自前 Docker 実行の違い
- CUDA 対応ライブラリの version 整合
  - PyTorch, torchvision, CUDA minor version の対応
  - 研究実装で壊れやすい native extension の依存関係
  - `xformers`, `mmcv`, `flash-attn`, custom op のような依存の扱い
- 推論高速化の基本
  - batch size と latency / throughput のトレードオフ
  - mixed precision, FP16, BF16
  - warmup, pinned memory, non-blocking transfer
  - 前処理と後処理のボトルネック切り分け
- エクスポートと最適化
  - ONNX export
  - ONNX Runtime による推論
  - TensorRT の基本
  - dynamic shape と static shape の違い
- GPU profiling の基礎
  - `nvidia-smi` の見方
  - メモリ使用量と OOM 原因切り分け
  - PyTorch profiler の初歩
  - CPU 待ちと GPU 待ちの見分け方
- 配備先ごとの最適化観点
  - SageMaker endpoint 向け推論
  - バッチ推論向け最適化
  - エッジ向け軽量化と runtime 選定

## 成果物

- `torch`, `torchvision`, CUDA version 対応表を含む GPU 環境セットアップノート
- 同一モデルに対する以下の比較レポート
  - CPU 推論
  - PyTorch GPU 推論
  - ONNX Runtime 推論
  - TensorRT 推論
- 前処理 / モデル / 後処理のどこが律速かを分離した profiling メモ
- SageMaker または Docker ベースの GPU 推論構成サンプル
- 論文実装を動かす際の依存関係チェックリスト

## 到達基準

- CUDA Toolkit を入れるべきケースと不要なケースを説明できる
- `nvidia-smi` が動くだけでは不十分である理由を説明できる
- `torch.cuda.is_available()` が失敗したときに、driver, container runtime, wheel のどこを疑うべきか切り分けできる
- 同一モデルで、精度を落とさずに速度改善できるポイントを複数挙げられる
- GPU 高速化の主因がモデル本体なのか、前後処理なのか、I/O なのかを切り分けできる
- 研究実装で CUDA extension build が必要になったとき、通常環境と分離して扱う判断ができる

## 最終成果物

- GPU 推論環境のセットアップ手順書
- 推論高速化ベンチマークテンプレート
- profiling 手順テンプレート
- 配備先別の実装指針メモ
  - SageMaker
  - 自前 Docker
  - エッジ
