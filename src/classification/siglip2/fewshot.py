from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
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
    build_hf_processor_collate_fn,
    build_label_mapping,
    collect_predictions,
    create_train_valid_split,
    fit_classifier,
    load_oxford_pet_test_dataframe,
    load_oxford_pet_trainval_dataframe,
    sample_fewshot_per_class,
)
from src.classification.siglip2.model import build_siglip2_model, build_siglip2_processor


DEFAULT_CONFIG_PATH = Path("/workspace/config/classification/siglip2_oxford_pet_fewshot.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train few-shot SigLIP 2 classifier.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    dataset_root: str,
    batch_size: int,
    num_workers: int,
    device: str,
    model_name: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_mapping = checkpoint["label_mapping"]
    processor = build_siglip2_processor(model_name)
    collate_fn = build_hf_processor_collate_fn(processor)

    test_df = load_oxford_pet_test_dataframe(dataset_root)
    test_dataset = OxfordIIITPetClassificationDataset(test_df, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_fn,
    )

    model = build_siglip2_model(num_classes=len(label_mapping), model_name=model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    prediction_df = collect_predictions(model, test_loader, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_dir / "predictions.csv", index=False)
    target_names = [label_mapping[index + 1] for index in range(len(label_mapping))]
    build_classification_report(prediction_df, target_names=target_names).to_csv(output_dir / "classification_report.csv")
    build_confusion_matrix(prediction_df, num_classes=len(label_mapping)).to_csv(output_dir / "confusion_matrix.csv", index=False)
    accuracy = float((prediction_df["label"] == prediction_df["prediction"]).mean())
    summary = {"num_test_examples": int(len(prediction_df)), "accuracy": accuracy, "checkpoint_path": str(checkpoint_path)}
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(raw["output_dir"])
    run_dir = output_dir / "fewshot"
    run_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_oxford_pet_trainval_dataframe(raw["dataset_root"])
    train_df, valid_df = create_train_valid_split(
        full_df,
        valid_size=raw["train"]["valid_size"],
        random_state=raw["train"]["random_state"],
    )
    train_df = sample_fewshot_per_class(
        train_df,
        shots_per_class=raw["fewshot"]["shots_per_class"],
        random_state=raw["fewshot"]["random_state"],
    )
    label_mapping = build_label_mapping(full_df)

    processor = build_siglip2_processor(raw["model_name"])
    collate_fn = build_hf_processor_collate_fn(processor)
    train_dataset = OxfordIIITPetClassificationDataset(train_df, transform=None)
    valid_dataset = OxfordIIITPetClassificationDataset(valid_df, transform=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=raw["train"]["batch_size"],
        shuffle=True,
        num_workers=raw["train"]["num_workers"],
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=raw["train"]["batch_size"],
        shuffle=False,
        num_workers=raw["train"]["num_workers"],
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_fn,
    )

    model = build_siglip2_model(num_classes=len(label_mapping), model_name=raw["model_name"]).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=raw["train"]["learning_rate"],
        weight_decay=raw["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=raw["train"]["use_amp"] and args.device.startswith("cuda"))
    best_valid_accuracy = -1.0

    run_config = {
        "experiment_name": raw["experiment_name"],
        "model_name": raw["model_name"],
        "dataset_root": raw["dataset_root"],
        "batch_size": raw["train"]["batch_size"],
        "num_workers": raw["train"]["num_workers"],
        "num_epochs": raw["train"]["num_epochs"],
        "learning_rate": raw["train"]["learning_rate"],
        "weight_decay": raw["train"]["weight_decay"],
        "shots_per_class": raw["fewshot"]["shots_per_class"],
        "fewshot_train_examples": int(len(train_df)),
        "run_name": "fewshot",
    }

    def epoch_end_callback(epoch: int, metrics: dict[str, float]) -> None:
        nonlocal best_valid_accuracy
        print(
            f"[fewshot] epoch={epoch} "
            f"train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_accuracy']:.4f} "
            f"valid_loss={metrics['valid_loss']:.4f} valid_acc={metrics['valid_accuracy']:.4f}"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": run_config,
                "label_mapping": label_mapping,
            },
            run_dir / "latest.pt",
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
                run_dir / "best.pt",
            )

    history = fit_classifier(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=raw["train"]["num_epochs"],
        device=args.device,
        scaler=scaler,
        epoch_end_callback=epoch_end_callback,
    )

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    save_json(run_dir / "config.json", run_config)
    save_json(run_dir / "label_mapping.json", {str(key): value for key, value in label_mapping.items()})

    test_summary = evaluate_checkpoint(
        checkpoint_path=run_dir / "best.pt",
        output_dir=output_dir / "best_test_results",
        dataset_root=raw["dataset_root"],
        batch_size=raw["train"]["batch_size"],
        num_workers=raw["train"]["num_workers"],
        device=args.device,
        model_name=raw["model_name"],
    )

    experiment_summary = {
        "experiment_name": raw["experiment_name"],
        "mode": "fewshot",
        "best_source": "fewshot",
        "best_run_name": "fewshot",
        "best_checkpoint_path": str(run_dir / "best.pt"),
        "best_valid_accuracy": float(best_valid_accuracy),
        "test_accuracy": test_summary["accuracy"],
        "model_name": raw["model_name"],
    }
    save_json(output_dir / "experiment_summary.json", experiment_summary)

    print("")
    print("few-shot training completed.")
    print(experiment_summary)


if __name__ == "__main__":
    main()
