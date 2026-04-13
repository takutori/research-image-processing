from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.common import (
    OxfordIIITPetClassificationDataset,
    build_classification_report,
    build_confusion_matrix,
    build_label_mapping,
    build_supervised_eval_transform,
    build_supervised_train_transform,
    collect_predictions,
    create_train_valid_split,
    expand_grid,
    fit_classifier,
    load_oxford_pet_test_dataframe,
    load_oxford_pet_trainval_dataframe,
    rank_tuning_results,
    save_tuning_artifacts,
)
from src.classification.convnext_v2.config import (
    DEFAULT_CONFIG_PATH,
    ConvNeXtV2ExperimentConfig,
    load_experiment_config,
)
from src.classification.convnext_v2.model import build_convnext_v2_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt V2 with config-driven pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_label_mapping_for_target_names(label_mapping: dict[int, str]) -> list[str]:
    return [label_mapping[index + 1] for index in range(len(label_mapping))]


def train_single_run(
    config: ConvNeXtV2ExperimentConfig,
    output_dir: Path,
    *,
    image_size: int,
    eval_resize_size: int,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    use_amp: bool,
    run_name: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_oxford_pet_trainval_dataframe(config.dataset_root)
    train_df, valid_df = create_train_valid_split(
        full_df,
        valid_size=config.train.valid_size,
        random_state=config.train.random_state,
    )
    label_mapping = build_label_mapping(full_df)

    train_dataset = OxfordIIITPetClassificationDataset(
        train_df,
        transform=build_supervised_train_transform(image_size=image_size),
    )
    valid_dataset = OxfordIIITPetClassificationDataset(
        valid_df,
        transform=build_supervised_eval_transform(
            image_size=image_size,
            resize_size=eval_resize_size,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = build_convnext_v2_model(
        num_classes=len(label_mapping),
        model_name=config.model_name,
        pretrained=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.startswith("cuda"))

    best_valid_accuracy = -1.0

    run_config = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "dataset_root": config.dataset_root,
        "image_size": image_size,
        "eval_resize_size": eval_resize_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "valid_size": config.train.valid_size,
        "random_state": config.train.random_state,
        "use_amp": use_amp,
        "run_name": run_name,
    }

    def epoch_end_callback(epoch: int, metrics: dict[str, float]) -> None:
        nonlocal best_valid_accuracy
        print(
            f"[{run_name}] epoch={epoch} "
            f"train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_accuracy']:.4f} "
            f"valid_loss={metrics['valid_loss']:.4f} valid_acc={metrics['valid_accuracy']:.4f}"
        )

        latest_checkpoint_path = output_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": run_config,
                "label_mapping": label_mapping,
            },
            latest_checkpoint_path,
        )

        if metrics["valid_accuracy"] > best_valid_accuracy:
            best_valid_accuracy = metrics["valid_accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": run_config,
                    "label_mapping": label_mapping,
                    "best_valid_accuracy": best_valid_accuracy,
                },
                output_dir / "best.pt",
            )

    history = fit_classifier(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        scaler=scaler,
        epoch_end_callback=epoch_end_callback,
    )

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    save_json(output_dir / "config.json", run_config)
    save_json(output_dir / "label_mapping.json", {str(key): value for key, value in label_mapping.items()})

    return {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "best_checkpoint_path": str(output_dir / "best.pt"),
        "best_valid_accuracy": float(best_valid_accuracy),
        "config": run_config,
        "label_mapping": label_mapping,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    dataset_root: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    label_mapping = checkpoint["label_mapping"]

    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = load_oxford_pet_test_dataframe(dataset_root)
    test_dataset = OxfordIIITPetClassificationDataset(
        test_df,
        transform=build_supervised_eval_transform(
            image_size=int(config["image_size"]),
            resize_size=int(config["eval_resize_size"]),
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = build_convnext_v2_model(
        num_classes=len(label_mapping),
        model_name=str(config["model_name"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    prediction_df = collect_predictions(model, test_loader, device=device)
    prediction_df.to_csv(output_dir / "predictions.csv", index=False)

    target_names = load_label_mapping_for_target_names(label_mapping)
    report_df = build_classification_report(prediction_df, target_names=target_names)
    report_df.to_csv(output_dir / "classification_report.csv")

    confusion_df = build_confusion_matrix(prediction_df, num_classes=len(label_mapping))
    confusion_df.to_csv(output_dir / "confusion_matrix.csv", index=False)

    accuracy = float((prediction_df["label"] == prediction_df["prediction"]).mean())
    summary = {
        "num_test_examples": int(len(prediction_df)),
        "accuracy": accuracy,
        "checkpoint_path": str(checkpoint_path),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def run_plain_training(config: ConvNeXtV2ExperimentConfig, device: str) -> dict[str, object]:
    train_dir = config.output_path / "train"
    train_result = train_single_run(
        config=config,
        output_dir=train_dir,
        image_size=config.train.image_size,
        eval_resize_size=config.train.eval_resize_size,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        num_epochs=config.train.num_epochs,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        device=device,
        use_amp=config.train.use_amp,
        run_name="train",
    )
    experiment_summary = {
        "experiment_name": config.experiment_name,
        "mode": "none",
        "best_source": "train",
        "best_run_name": "train",
        "best_checkpoint_path": train_result["best_checkpoint_path"],
        "best_valid_accuracy": train_result["best_valid_accuracy"],
    }
    save_json(config.output_path / "experiment_summary.json", experiment_summary)
    return experiment_summary


def run_grid_tuning(config: ConvNeXtV2ExperimentConfig, device: str) -> pd.DataFrame:
    tuning_root = config.output_path / "grid"
    param_grid = {
        "learning_rate": config.grid.learning_rates,
        "weight_decay": config.grid.weight_decays,
        "batch_size": config.grid.batch_sizes,
        "image_size": config.grid.image_sizes,
    }
    trials = expand_grid(param_grid)

    trial_rows: list[dict[str, object]] = []
    for trial_index, trial_params in enumerate(trials, start=1):
        trial_name = f"trial_{trial_index:03d}"
        train_result = train_single_run(
            config=config,
            output_dir=tuning_root / trial_name,
            image_size=int(trial_params["image_size"]),
            eval_resize_size=int(trial_params["image_size"]) + 32,
            batch_size=int(trial_params["batch_size"]),
            num_workers=config.grid.num_workers,
            num_epochs=config.grid.num_epochs,
            learning_rate=float(trial_params["learning_rate"]),
            weight_decay=float(trial_params["weight_decay"]),
            device=device,
            use_amp=config.train.use_amp,
            run_name=trial_name,
        )
        trial_rows.append(
            {
                "trial_name": trial_name,
                "output_dir": train_result["output_dir"],
                "best_checkpoint_path": train_result["best_checkpoint_path"],
                "best_valid_accuracy": train_result["best_valid_accuracy"],
                **trial_params,
            }
        )

    results_df = rank_tuning_results(pd.DataFrame(trial_rows))
    save_tuning_artifacts(
        tuning_root,
        results_df,
        metadata={
            "experiment_name": config.experiment_name,
            "model_name": config.model_name,
            "dataset_root": config.dataset_root,
            "device": device,
            "param_grid": param_grid,
        },
    )
    return results_df


def run_optuna_tuning(config: ConvNeXtV2ExperimentConfig, device: str) -> pd.DataFrame:
    output_dir = config.output_path / "optuna"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_grid_trial = pd.read_csv(config.output_path / "grid" / "tuning_results.csv").iloc[0].to_dict()
    best_lr = float(best_grid_trial["learning_rate"])
    best_weight_decay = float(best_grid_trial["weight_decay"])
    best_batch_size = int(best_grid_trial["batch_size"])
    best_image_size = int(best_grid_trial["image_size"])

    batch_size_candidates = config.optuna.batch_size_candidates or sorted(
        {max(8, best_batch_size // 2), best_batch_size, best_batch_size * 2}
    )
    image_size_candidates = config.optuna.image_size_candidates or [best_image_size]

    lr_low, lr_high = best_lr / 3.0, best_lr * 3.0
    wd_low, wd_high = best_weight_decay / 5.0, best_weight_decay * 5.0

    records: list[dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
        weight_decay = trial.suggest_float("weight_decay", wd_low, wd_high, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_size_candidates)
        image_size = trial.suggest_categorical("image_size", image_size_candidates)

        trial_name = f"trial_{trial.number:03d}"
        train_result = train_single_run(
            config=config,
            output_dir=output_dir / trial_name,
            image_size=image_size,
            eval_resize_size=image_size + 32,
            batch_size=batch_size,
            num_workers=config.optuna.num_workers,
            num_epochs=config.optuna.num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            use_amp=config.train.use_amp,
            run_name=trial_name,
        )

        records.append(
            {
                "trial_name": trial_name,
                "output_dir": train_result["output_dir"],
                "best_checkpoint_path": train_result["best_checkpoint_path"],
                "best_valid_accuracy": train_result["best_valid_accuracy"],
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "image_size": image_size,
            }
        )
        return float(train_result["best_valid_accuracy"])

    study = optuna.create_study(direction="maximize", study_name=f"{config.experiment_name}_optuna")
    study.optimize(objective, n_trials=config.optuna.n_trials)

    results_df = rank_tuning_results(pd.DataFrame(records))
    results_df.to_csv(output_dir / "optuna_results.csv", index=False)
    save_json(
        output_dir / "optuna_summary.json",
        {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "grid_best_trial": best_grid_trial,
            "search_space": {
                "learning_rate": [lr_low, lr_high],
                "weight_decay": [wd_low, wd_high],
                "batch_size_candidates": batch_size_candidates,
                "image_size_candidates": image_size_candidates,
            },
        },
    )
    return results_df


def run_full_tuning(config: ConvNeXtV2ExperimentConfig, device: str) -> dict[str, object]:
    run_grid_tuning(config, device)
    optuna_df = run_optuna_tuning(config, device)

    best_trial = optuna_df.iloc[0].to_dict()
    best_checkpoint_path = Path(best_trial["best_checkpoint_path"])

    experiment_summary = {
        "experiment_name": config.experiment_name,
        "mode": "full",
        "best_source": "optuna",
        "best_run_name": best_trial["trial_name"],
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_valid_accuracy": float(best_trial["best_valid_accuracy"]),
        "best_trial": best_trial,
    }
    save_json(config.output_path / "experiment_summary.json", experiment_summary)
    return experiment_summary


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    if config.tuning_mode == "none":
        summary = run_plain_training(config, args.device)
    elif config.tuning_mode == "grid":
        run_grid_tuning(config, args.device)
        summary = {"mode": "grid"}
    elif config.tuning_mode == "optuna":
        run_optuna_tuning(config, args.device)
        summary = {"mode": "optuna"}
    elif config.tuning_mode == "full":
        summary = run_full_tuning(config, args.device)
    else:
        raise ValueError(f"unsupported tuning_mode: {config.tuning_mode}")

    print("")
    print("pipeline completed.")
    print(summary)


if __name__ == "__main__":
    main()
