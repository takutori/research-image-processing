from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.tracking.common.data import SequenceInfo


def require_trackeval():
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    try:
        import trackeval
    except ImportError as error:
        raise ModuleNotFoundError(
            "Tracking evaluation requires `trackeval` at runtime. Run `uv sync` before executing tracking scripts."
        ) from error
    return trackeval


def _metric_value(metric_dict: dict, *candidates: str) -> float | int:
    for key in candidates:
        if key in metric_dict:
            return metric_dict[key]
    raise KeyError(f"metric key not found. tried={candidates}, available={sorted(metric_dict.keys())}")


def save_mot_predictions(
    output_dir: str | Path,
    tracker_name: str,
    prediction_rows: list[dict],
) -> tuple[pd.DataFrame, Path, Path]:
    output_dir = Path(output_dir)
    best_dir = output_dir / "best_test_results"
    best_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(prediction_rows)
    if predictions_df.empty:
        predictions_df = pd.DataFrame(columns=["sequence", "frame", "track_id", "x", "y", "w", "h", "score"])
    predictions_df = predictions_df.sort_values(["sequence", "frame", "track_id"]).reset_index(drop=True)
    predictions_df.to_csv(best_dir / "predictions.csv", index=False)

    mot_root = best_dir / "mot_txt"
    mot_root.mkdir(parents=True, exist_ok=True)
    trackeval_tracker_root = best_dir / "trackers" / tracker_name / "data"
    trackeval_tracker_root.mkdir(parents=True, exist_ok=True)
    if predictions_df.empty:
        return predictions_df, mot_root, trackeval_tracker_root

    for sequence_name, seq_df in predictions_df.groupby("sequence"):
        export_df = seq_df.loc[:, ["frame", "track_id", "x", "y", "w", "h", "score"]].assign(
            x_world=-1,
            y_world=-1,
            z_world=-1,
        )
        export_df.to_csv(
            mot_root / f"{sequence_name}.txt",
            header=False,
            index=False,
        )
        export_df.to_csv(
            trackeval_tracker_root / f"{sequence_name}.txt",
            header=False,
            index=False,
        )
    return predictions_df, mot_root, trackeval_tracker_root


def evaluate_mot_tracking(
    eval_gt_root: str | Path,
    sequences: list[SequenceInfo],
    trackers_root: str | Path,
    tracker_name: str,
    output_root: str | Path,
    frame_limit: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    trackeval = require_trackeval()
    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config["PRINT_CONFIG"] = False
    evaluator = trackeval.Evaluator(evaluator_config)

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(eval_gt_root),
            "TRACKERS_FOLDER": str(trackers_root),
            "OUTPUT_FOLDER": str(output_root),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "trackeval",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {
                sequence.name: min(sequence.seq_length, frame_limit) if frame_limit else sequence.seq_length
                for sequence in sequences
            },
            "PRINT_CONFIG": False,
        }
    )
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]

    evaluation_result = evaluator.evaluate([dataset], metrics_list)
    output_res = evaluation_result[0] if isinstance(evaluation_result, tuple) else evaluation_result

    tracker_res = output_res["MotChallenge2DBox"][tracker_name]
    combined = tracker_res["COMBINED_SEQ"]["pedestrian"]

    overall = {
        "tracker_name": tracker_name,
        "HOTA": float(np.mean(combined["HOTA"]["HOTA"])),
        "DetA": float(np.mean(combined["HOTA"]["DetA"])),
        "AssA": float(np.mean(combined["HOTA"]["AssA"])),
        "IDF1": float(combined["Identity"]["IDF1"]),
        "MOTA": float(combined["CLEAR"]["MOTA"]),
        "FP": int(_metric_value(combined["CLEAR"], "FP", "CLR_FP")),
        "FN": int(_metric_value(combined["CLEAR"], "FN", "CLR_FN")),
        "IDs": int(_metric_value(combined["CLEAR"], "IDSW", "IDs")),
    }

    sequence_rows: list[dict] = []
    for sequence in sequences:
        seq_res = tracker_res[sequence.name]["pedestrian"]
        sequence_rows.append(
            {
                "sequence": sequence.name,
                "HOTA": float(np.mean(seq_res["HOTA"]["HOTA"])),
                "DetA": float(np.mean(seq_res["HOTA"]["DetA"])),
                "AssA": float(np.mean(seq_res["HOTA"]["AssA"])),
                "IDF1": float(seq_res["Identity"]["IDF1"]),
                "MOTA": float(seq_res["CLEAR"]["MOTA"]),
                "FP": int(_metric_value(seq_res["CLEAR"], "FP", "CLR_FP")),
                "FN": int(_metric_value(seq_res["CLEAR"], "FN", "CLR_FN")),
                "IDs": int(_metric_value(seq_res["CLEAR"], "IDSW", "IDs")),
            }
        )
    sequence_metrics = pd.DataFrame(sequence_rows).sort_values("sequence").reset_index(drop=True)
    return overall, sequence_metrics


def save_tracking_reports(
    output_dir: str | Path,
    overall_metrics: dict,
    sequence_metrics: pd.DataFrame,
    gt_stats: pd.DataFrame,
    selected_sequences: dict[str, str],
    experiment_name: str,
) -> None:
    output_dir = Path(output_dir)
    best_dir = output_dir / "best_test_results"
    best_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "experiment_name": experiment_name,
        **overall_metrics,
        "selected_sequences": selected_sequences,
    }
    (best_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    sequence_metrics.to_csv(best_dir / "sequence_metrics.csv", index=False)
    gt_stats.to_csv(best_dir / "gt_sequence_stats.csv", index=False)
    (best_dir / "selected_sequences.json").write_text(
        json.dumps(selected_sequences, ensure_ascii=False, indent=2)
    )


def select_representative_sequences(
    sequence_metrics: pd.DataFrame,
    gt_stats: pd.DataFrame,
) -> dict[str, str]:
    if sequence_metrics.empty:
        return {}

    merged = sequence_metrics.merge(gt_stats, on="sequence", how="left")
    selected: dict[str, str] = {}
    used: set[str] = set()

    def choose(order_df: pd.DataFrame) -> str | None:
        for sequence_name in order_df["sequence"].tolist():
            if sequence_name not in used:
                used.add(sequence_name)
                return sequence_name
        return None

    best = choose(merged.sort_values(["HOTA", "IDF1", "sequence"], ascending=[False, False, True]))
    if best:
        selected["best"] = best

    worst = choose(merged.sort_values(["HOTA", "IDF1", "sequence"], ascending=[True, True, True]))
    if worst:
        selected["worst"] = worst

    median_hota = merged["HOTA"].median()
    middle = choose(
        merged.assign(median_distance=(merged["HOTA"] - median_hota).abs()).sort_values(
            ["median_distance", "sequence"]
        )
    )
    if middle:
        selected["middle"] = middle

    hard = choose(
        merged.sort_values(["avg_objects_per_frame", "max_objects_per_frame", "sequence"], ascending=[False, False, True])
    )
    if hard:
        selected["hard"] = hard

    return selected
