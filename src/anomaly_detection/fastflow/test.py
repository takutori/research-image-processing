from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.common import update_summary_with_test_metrics
from src.anomaly_detection.fastflow.config import DEFAULT_CONFIG_PATH, load_experiment_config
from src.anomaly_detection.fastflow.train import run_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best FastFlow checkpoint on test set.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    summary_path = config.output_path / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"experiment_summary.json not found: {summary_path}. Run train.py first.")
    summary = json.loads(summary_path.read_text())
    best_artifact_path = Path(summary["best_artifact_path"])
    result = run_test(config, best_artifact_path, model_params=summary.get("best_model_config"))
    update_summary_with_test_metrics(
        summary_path,
        test_image_level_metric=result["image_auroc"],
        test_pixel_level_metric=result["pixel_auroc"],
    )
    print(f"test completed. results saved to: {config.output_path / 'best_test_results'}")


if __name__ == "__main__":
    main()
