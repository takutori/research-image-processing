from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def expand_grid(param_grid: dict[str, Iterable[object]]) -> list[dict[str, object]]:
    keys = list(param_grid.keys())
    values = [list(param_grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def rank_tuning_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if len(results_df) == 0:
        return results_df
    sort_columns = [column for column in ["best_valid_accuracy", "test_accuracy"] if column in results_df.columns]
    if not sort_columns:
        return results_df
    return results_df.sort_values(sort_columns, ascending=False).reset_index(drop=True)


def save_tuning_artifacts(output_dir: Path, results_df: pd.DataFrame, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "tuning_results.csv", index=False)
    (output_dir / "tuning_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )


def load_best_trial(results_csv_path: Path | str) -> dict[str, object]:
    results_csv_path = Path(results_csv_path)
    if not results_csv_path.exists():
        raise FileNotFoundError(f"tuning results not found: {results_csv_path}")

    results_df = pd.read_csv(results_csv_path)
    ranked_df = rank_tuning_results(results_df)
    if len(ranked_df) == 0:
        raise ValueError(f"tuning results are empty: {results_csv_path}")

    return ranked_df.iloc[0].to_dict()
