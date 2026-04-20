from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Florence2TrainConfig:
    image_size: int = 224
    eval_resize_size: int = 224
    batch_size: int = 8
    num_workers: int = 0
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    valid_size: float = 0.2
    random_state: int = 42
    use_amp: bool = True


@dataclass
class Florence2GridConfig:
    learning_rates: list[float]
    weight_decays: list[float]
    batch_sizes: list[int]
    image_sizes: list[int]
    num_epochs: int = 2
    num_workers: int = 0


@dataclass
class Florence2OptunaConfig:
    n_trials: int = 10
    num_epochs: int = 2
    num_workers: int = 0
    batch_size_candidates: list[int] | None = None
    image_size_candidates: list[int] | None = None


@dataclass
class Florence2SubsetConfig:
    train_examples: int | None = None
    valid_examples: int | None = None
    test_examples: int | None = None


@dataclass
class Florence2ExperimentConfig:
    experiment_name: str
    model_name: str
    dataset_root: str
    output_dir: str
    tuning_mode: str
    train: Florence2TrainConfig
    grid: Florence2GridConfig
    optuna: Florence2OptunaConfig
    subset: Florence2SubsetConfig | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path("/workspace/config/classification/florence2_oxford_pet.yaml")


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Florence2ExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())
    return Florence2ExperimentConfig(
        experiment_name=raw["experiment_name"],
        model_name=raw["model_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        tuning_mode=raw.get("tuning_mode", "full"),
        train=Florence2TrainConfig(**raw["train"]),
        grid=Florence2GridConfig(**raw["grid"]),
        optuna=Florence2OptunaConfig(**raw["optuna"]),
        subset=Florence2SubsetConfig(**raw["subset"]) if raw.get("subset") else None,
    )
