from __future__ import annotations

from pathlib import Path

from src.tracking.common.config import (
    TrackingExperimentConfig,
    load_tracking_experiment_config,
)


DEFAULT_CONFIG_PATH = Path("/workspace/config/tracking/bytetrack_dancetrack_val.yaml")


def load_experiment_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> TrackingExperimentConfig:
    return load_tracking_experiment_config(config_path)
