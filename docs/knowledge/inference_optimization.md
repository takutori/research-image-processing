# Inference Optimization for Anomaly Detection

## Scope

対象は学習済みモデルの推論速度最適化です。再学習が必要な変更は除外します。

対象モデル:

- PatchCore
- FastFlow
- PaDiM
- WinCLIP zero-shot

比較したいもの:

- 最適化なしの推論部分の計算時間
- 最適化ありの推論部分の計算時間
- 同じ test set での評価指標の変化

評価 artifact 作成、CSV 保存、可視化画像保存、notebook 描画の速度は対象外です。

## Applicability Matrix

凡例:

- `○`: 再学習なしで適用しやすい
- `△`: 再学習なしで試せるが、評価値や数値安定性が変わる可能性がある
- `-`: 対象外

| 推論最適化アプローチ | PatchCore | FastFlow | PaDiM | WinCLIP zero-shot | 備考 |
|---|---:|---:|---:|---:|---|
| 推論専用 runner 化 | ○ | ○ | ○ | ○ | `Engine.predict` ではなく、ロード済みモデルに対して推論部分だけを測る |
| `eval()` / `torch.inference_mode()` 徹底 | ○ | ○ | ○ | ○ | 勾配計算と training 挙動を完全に切る |
| artifact の起動時常駐ロード | ○ | ○ | ○ | ○ | checkpoint、統計量、memory bank、text embedding をリクエストごとに読まない |
| batch 推論サイズ調整 | ○ | ○ | ○ | ○ | throughput 測定では必須。latency 測定では batch size 1 も別途見る |
| GPU 常駐化 | ○ | ○ | ○ | ○ | モデル、統計量、memory bank、中間 tensor を不要に CPU へ戻さない |
| CPU/GPU 転送削減 | ○ | ○ | ○ | ○ | 最終 score だけ必要なら map や中間 feature の転送を避ける |
| image score のみ計算 | △ | △ | △ | △ | heatmap 不要の運用なら候補。pixel 指標や UI 用 heatmap は失う |
| mixed precision 推論 | △ | ○ | △ | ○ | FastFlow / WinCLIP は有力。PatchCore / PaDiM は距離計算の数値影響を確認する |
| `torch.compile` | △ | ○ | △ | △ | PyTorch 2 系で候補。初回 compile 時間を除外して測る |
| ONNX / TensorRT export | △ | ○ | △ | ○ | 効果は大きいが、postprocess や特殊演算の切り分けが必要 |
| memory bank GPU 常駐 | ○ | - | - | - | PatchCore の近傍探索で CPU/GPU 往復を避ける |
| memory bank `float16` 化 | △ | - | - | - | PatchCore のメモリ帯域削減。距離順序が変わる可能性あり |
| FAISS / 高速近傍探索 | ○ | - | - | - | PatchCore の主要ボトルネックに直接効く |
| 統計量 GPU 常駐 | - | - | ○ | - | PaDiM の平均・共分散逆行列を GPU に置く |
| Mahalanobis 距離の vectorized torch 実装 | - | - | ○ | - | PaDiM の Python loop を避ける |
| 統計量 `float16` 化 | - | - | △ | - | PaDiM のメモリ削減候補。Mahalanobis 距離の数値影響を確認する |
| text embedding cache | - | - | - | ○ | WinCLIP の prompt tokenize / text encode を推論ごとに行わない |
| text encoder 実行の除去 | - | - | - | ○ | cached text embedding だけを使う |
| image encoder mixed precision | - | - | - | ○ | CLIP image encoder の GPU 推論に効きやすい |
| image encoder export | - | - | - | ○ | CLIP image encoder を ONNX / TensorRT 化する候補 |

## Excluded From This Scope

以下は速度には効く可能性がありますが、学習条件またはモデル条件を変えるため、この検討から外します。

| 除外する変更 | 理由 |
|---|---|
| 入力解像度変更 | score 分布と threshold が変わる。実質的に別条件の評価になる |
| backbone 変更 | checkpoint と構造が合わないため再学習が必要 |
| FastFlow `flow_steps` / `hidden_ratio` 変更 | モデル構造が変わるため再学習が必要 |
| PaDiM `n_features` 変更 | 学習済み統計量の次元と対応しない |
| PatchCore `coreset_sampling_ratio` を学習設定として変更 | memory bank 構築条件が変わる |
| WinCLIP prompt 数削減 | zero-shot のモデル条件を変える。再学習は不要だが評価条件が変わるため別実験扱い |

## Recommended Order

まずは全モデルに共通して適用でき、評価値を変えにくいものから測るのがよいです。

1. 推論専用 runner 化
2. artifact の起動時常駐ロード
3. `eval()` / `torch.inference_mode()` 徹底
4. batch 推論サイズ調整
5. GPU 常駐化と CPU/GPU 転送削減
6. モデル固有の高速化

モデル固有の優先候補:

| モデル | 優先候補 |
|---|---|
| PatchCore | memory bank GPU 常駐、FAISS / 高速近傍探索 |
| FastFlow | mixed precision、`torch.compile`、export |
| PaDiM | 統計量 GPU 常駐、Mahalanobis 距離の vectorized torch 実装 |
| WinCLIP zero-shot | text embedding cache、text encoder 実行の除去、image encoder mixed precision |

