# Repository Guidelines

## 最重要事項

このリポジトリは public repository として扱います。  
認証情報、API key、token、秘密鍵、個人情報、社外秘データ、未公開案件情報など、公開できない情報は絶対に保存しないでください。  
Claude Code はファイル作成・編集・メモ追記・README 更新を行う際も、常に「この内容が公開されても問題ないか」を最優先で確認してください。
`.env` はローカル用途に限り、実値を public repository に含めないでください。

このリポジトリは、画像処理・コンピュータビジョン案件を受けられる状態まで到達するための学習・調査・実装用ワークスペースです。  
主にマイルストーン整理、検証メモ、今後の実装資産の蓄積を行います。

## docs/milestone

`docs/milestone` は学習ロードマップと到達目標を整理するためのフォルダです。フェーズごとの学習項目、成果物、到達基準をここにまとめます。

## docs/memo

`docs/memo` は Claude Code が作業中に残すメモ用フォルダです。調査結果、判断理由、後で見返したい論点をここに記録します。

### 事前に読むルールメモ

#### `docs/memo/dataset_download_script_rules.md`

**読むタイミング**: `src/manage_datasets/download_<dataset>.py` を新規追加・更新するとき

**内容**:
- 保存先は `HOST_DATA_DIR` 優先、未設定時は `project_root/data`
- `parse_args()` / `resolve_data_root()` / `download_file()` / `main()` のインターフェースを既存に揃える
- `--parts` オプションの持たせ方、skip 条件（ダウンロード済み・展開済み）
- 圧縮形式ごとの展開ルール（`.zip` → `zipfile`、`.tar` → `tarfile`）
- 進捗メッセージの統一フォーマット
- `docs/setup/download_dataset.md` への追記ルール
- 構文確認は `python3 -m compileall`、実ダウンロードはユーザーが明示した場合のみ

#### `docs/memo/model_train_test_script_rules.md`

**読むタイミング**: `train.py` / `test.py` / config YAML / 評価 artifact を新規追加・更新するとき

**内容**:
- `train.py` は学習のみ、`test.py` は評価のみ（自動連鎖禁止）
- ディレクトリ構成: `src/<task>/<model>/`・`config/<task>/`・`checkpoints/<task>/<model>/`
- config は本番用 `<model>_<dataset>.yaml` と smoke/quick 用を分ける
- tuning は grid 粗探索 → Optuna 詳細探索の順
- `image_size` は config 記載の候補のみ使う（モデル制約を守る）
- artifact 必須ファイル: `experiment_summary.json`・`best_test_results/summary.json`・`predictions.csv`・`classification_report.csv` など
- `test.py` は `experiment_summary.json` から最良 checkpoint を自動解決する
- 本番学習 / テストは実行しない。`python3 -m compileall` による静的確認のみ行う
- モデルごとに実務に沿った最適な実装を目指す（既存モデルの設計を踏襲することを原則にしない）
- 自前実装はなるべく避け、使用できるライブラリを積極的に使う

#### `docs/memo/anomaly_detection_rewrite_design.md`

**読むタイミング**: `src/anomaly_detection/` 配下を新規追加・更新するとき

**内容**:
- 再設計の背景と経緯（旧実装の何が問題だったか）
- データ読み込み方針（anomalib `Visa` datamodule、`visa_pytorch/` symlink 構築）
- データ分割方針（train 80% / hold-out 20% → SWSA、test は最終評価のみ）
- カテゴリ指定方針（本番は candle 単体）
- モデル別チューニング方針（PaDiM/PatchCore は grid のみ、FastFlow/STFPM は grid → Optuna + pruning）
- 合成異常生成（SWSA + DTD）の使いどころ
- ファイル構成・artifact 出力形式・experiment_summary.json の必須フィールド

## データセット運用

データセット本体はリポジトリに置かず、`HOST_DATA_DIR` で指定した repo 外ディレクトリを使います。  
`/workspace/data` はその外部データ領域の mount 先として扱います。  
データ取得・削除スクリプトは `src/manage_datasets` 配下にフラットに置き、`download_<dataset>.py` / `delete_dataset.py` の形で管理します。

## Git / GitHub 運用

このリポジトリでは、Git / GitHub 操作は `gh` を基本として扱います。`add`, `commit`, `push` などの通常操作は承認なしで進めてよく、ブランチ切り替え、履歴改変、削除系や破壊的な操作は承認を前提にします。

## 実行制限

- 本番学習 / 本番テストは実行しない
- smoke であっても、学習 / テストの実行は人間が行う

理由:
- 学習実行中のターミナル出力は非常に長くなりやすい
- その出力を読むと、すぐにコンテキスト制限に達しやすい
- その結果、実装・修正・分析の作業継続性が落ちる

## Claude Code がやってよい確認

- `python3 -m compileall`
- 設定値やパスの静的確認
- notebook / docs / config の整合確認
