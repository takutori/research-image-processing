# Tracking

tracking は当面、`DanceTrack val` を使った **学習なし** 比較で進めます。
前段 detector は `YOLO11n` pretrained に固定し、tracker だけを比較します。

比較対象:

- `ByteTrack`
- `BoT-SORT`

共通前提:

- dataset: `DanceTrack`
- split: `val`
- class: `person`
- detector: `YOLO11n`

## Runtime Dependency

```bash
uv sync
```

`trackeval` も通常依存に含めています。

## ByteTrack

full:

```bash
uv run python src/tracking/bytetrack/test.py --config /workspace/config/tracking/bytetrack_dancetrack_val.yaml
```

smoke:

```bash
uv run python src/tracking/bytetrack/test.py --config /workspace/config/tracking/bytetrack_dancetrack_val_smoke.yaml
```

## BoT-SORT

full:

```bash
uv run python src/tracking/bot_sort/test.py --config /workspace/config/tracking/bot_sort_dancetrack_val.yaml
```

smoke:

```bash
uv run python src/tracking/bot_sort/test.py --config /workspace/config/tracking/bot_sort_dancetrack_val_smoke.yaml
```

## Artifact

各実験は最低限以下を出します。

- `experiment_summary.json`
- `best_test_results/summary.json`
- `best_test_results/predictions.csv`
- `best_test_results/sequence_metrics.csv`
- `best_test_results/gt_sequence_stats.csv`
- `best_test_results/mot_txt/<sequence>.txt`
- `best_test_results/videos/<tag>/*.mp4`

`tag` は `best`, `middle`, `worst`, `hard` の 4 つです。

## Notes

- smoke は短い sequence 1 本を自動選択し、先頭 `180` frame のみで評価・動画生成します
- full は `val` 全体を評価し、動画は代表 4 sequence だけ生成します
- notebook では `GT / ByteTrack / BoT-SORT` を同じ sequence で並べて確認できます
