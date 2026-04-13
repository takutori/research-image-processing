from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ConvNeXtV2TrainConfig:
    image_size: int = 224
    eval_resize_size: int = 256
    batch_size: int = 32
    num_workers: int = 0
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    valid_size: float = 0.2
    random_state: int = 42
    use_amp: bool = True


@dataclass
class ConvNeXtV2GridConfig:
    learning_rates: list[float]
    weight_decays: list[float]
    batch_sizes: list[int]
    image_sizes: list[int]
    num_epochs: int = 3
    num_workers: int = 0


@dataclass
class ConvNeXtV2OptunaConfig:
    n_trials: int = 10
    num_epochs: int = 3
    num_workers: int = 0
    batch_size_candidates: list[int] | None = None
    image_size_candidates: list[int] | None = None


@dataclass
class ConvNeXtV2ExperimentConfig:
    experiment_name: str
    model_name: str
    dataset_root: str
    output_dir: str
    tuning_mode: str
    train: ConvNeXtV2TrainConfig
    grid: ConvNeXtV2GridConfig
    optuna: ConvNeXtV2OptunaConfig

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path("/workspace/config/classification/convnext_v2_oxford_pet.yaml")


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ConvNeXtV2ExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    train_cfg = ConvNeXtV2TrainConfig(**raw["train"])
    grid_cfg = ConvNeXtV2GridConfig(**raw["grid"])
    optuna_cfg = ConvNeXtV2OptunaConfig(**raw["optuna"])

    return ConvNeXtV2ExperimentConfig(
        experiment_name=raw["experiment_name"],
        model_name=raw["model_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        tuning_mode=raw.get("tuning_mode", "full"),
        train=train_cfg,
        grid=grid_cfg,
        optuna=optuna_cfg,
    )
