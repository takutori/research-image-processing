from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.bytetrack.config import DEFAULT_CONFIG_PATH, load_experiment_config
from src.tracking.bytetrack.model import run_bytetrack_on_sequence
from src.tracking.common.data import (
    build_gt_stats,
    list_split_sequences,
    resolve_dancetrack_root,
    select_shortest_sequences,
    write_trackeval_gt_root,
)
from src.tracking.common.evaluation import (
    evaluate_mot_tracking,
    save_mot_predictions,
    save_tracking_reports,
    select_representative_sequences,
)
from src.tracking.common.visualization import generate_experiment_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO11n + ByteTrack on DanceTrack val.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    dataset_root = resolve_dancetrack_root(config.dataset_root)
    sequences = list_split_sequences(dataset_root, config.split, sequence_names=config.test.sequence_names)
    if config.test.auto_select_shortest_n:
        sequences = select_shortest_sequences(sequences, config.test.auto_select_shortest_n)

    prediction_rows: list[dict] = []
    for index, sequence in enumerate(sequences, start=1):
        print(f"[{index}/{len(sequences)}] tracking {sequence.name}")
        prediction_rows.extend(
            run_bytetrack_on_sequence(
                sequence=sequence,
                detector_model=config.test.detector_model,
                tracker_name=config.test.tracker_name,
                img_size=config.test.img_size,
                conf=config.test.conf,
                iou=config.test.iou,
                device=config.test.device,
                frame_limit=config.test.frame_limit,
            )
        )

    predictions_df, mot_root, trackeval_tracker_root = save_mot_predictions(
        config.output_path,
        "bytetrack",
        prediction_rows,
    )
    eval_gt_root = write_trackeval_gt_root(
        sequences,
        config.output_path / "best_test_results" / "trackeval_gt",
        frame_limit=config.test.frame_limit,
    )
    overall_metrics, sequence_metrics = evaluate_mot_tracking(
        eval_gt_root=eval_gt_root,
        sequences=sequences,
        trackers_root=trackeval_tracker_root.parents[1],
        tracker_name="bytetrack",
        output_root=config.output_path / "best_test_results" / "trackeval_output",
        frame_limit=config.test.frame_limit,
    )

    gt_stats = build_gt_stats(sequences, frame_limit=config.test.frame_limit)
    selected_sequences = select_representative_sequences(sequence_metrics, gt_stats)
    save_tracking_reports(
        output_dir=config.output_path,
        overall_metrics=overall_metrics,
        sequence_metrics=sequence_metrics,
        gt_stats=gt_stats,
        selected_sequences=selected_sequences,
        experiment_name=config.experiment_name,
    )
    generate_experiment_videos(
        sequences=sequences,
        selected_sequences=selected_sequences,
        mot_txt_root=mot_root,
        output_dir=config.output_path,
        frame_limit=config.test.visualization_frame_limit,
        tracker_video_name="bytetrack",
    )

    summary = {
        "experiment_name": config.experiment_name,
        "mode": "zeroshot",
        "best_source": "zeroshot",
        "best_run_name": "zeroshot",
        "best_checkpoint_path": config.test.detector_model,
        "best_valid_hota": overall_metrics["HOTA"],
        "test_hota": overall_metrics["HOTA"],
        "test_idf1": overall_metrics["IDF1"],
        "test_mota": overall_metrics["MOTA"],
        "test_ids": overall_metrics["IDs"],
        "test_fp": overall_metrics["FP"],
        "test_fn": overall_metrics["FN"],
        "model_config": config.to_dict()["test"],
        "selected_sequences": selected_sequences,
        "num_sequences": len(sequences),
        "num_predictions": int(len(predictions_df)),
    }
    summary_path = config.output_path / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"test completed. results saved to: {config.output_path / 'best_test_results'}")


if __name__ == "__main__":
    main()
