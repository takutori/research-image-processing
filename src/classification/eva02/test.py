from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.eva02.config import DEFAULT_CONFIG_PATH, load_experiment_config
from src.classification.eva02.train import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best EVA-02 checkpoint from config.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    summary_path = config.output_path / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"experiment summary not found: {summary_path}. "
            "Run train.py with this config first."
        )

    summary = json.loads(summary_path.read_text())
    checkpoint_path = Path(summary["best_checkpoint_path"])
    test_summary = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        output_dir=config.output_path / "best_test_results",
        dataset_root=config.dataset_root,
        batch_size=int(summary.get("best_trial", {}).get("batch_size", config.train.batch_size)),
        num_workers=config.optuna.num_workers,
        device=device,
    )
    summary["test_accuracy"] = test_summary["accuracy"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"test completed. accuracy={test_summary['accuracy']:.4f}")
    print(f"results saved to: {config.output_path / 'best_test_results'}")


if __name__ == "__main__":
    main()
