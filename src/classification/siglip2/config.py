from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class SigLIP2TrainConfig:
    image_size: int = 224
    eval_resize_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    valid_size: float = 0.2
    random_state: int = 42
    use_amp: bool = True


@dataclass
class SigLIP2GridConfig:
    learning_rates: list[float]
    weight_decays: list[float]
    batch_sizes: list[int]
    image_sizes: list[int]
    num_epochs: int = 3
    num_workers: int = 0


@dataclass
class SigLIP2OptunaConfig:
    n_trials: int = 10
    num_epochs: int = 3
    num_workers: int = 0
    batch_size_candidates: list[int] | None = None
    image_size_candidates: list[int] | None = None


@dataclass
class SigLIP2ExperimentConfig:
    experiment_name: str
    model_name: str
    dataset_root: str
    output_dir: str
    tuning_mode: str
    train: SigLIP2TrainConfig
    grid: SigLIP2GridConfig
    optuna: SigLIP2OptunaConfig

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


DEFAULT_CONFIG_PATH = Path("/workspace/config/classification/siglip2_oxford_pet.yaml")


def load_experiment_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> SigLIP2ExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())

    return SigLIP2ExperimentConfig(
        experiment_name=raw["experiment_name"],
        model_name=raw["model_name"],
        dataset_root=raw["dataset_root"],
        output_dir=raw["output_dir"],
        tuning_mode=raw.get("tuning_mode", "full"),
        train=SigLIP2TrainConfig(**raw["train"]),
        grid=SigLIP2GridConfig(**raw["grid"]),
        optuna=SigLIP2OptunaConfig(**raw["optuna"]),
    )

