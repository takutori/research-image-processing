from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.common import (
    OxfordIIITPetClassificationDataset,
    build_classification_report,
    build_confusion_matrix,
    build_hf_processor_collate_fn,
    build_label_mapping,
    create_train_valid_split,
    load_oxford_pet_test_dataframe,
    load_oxford_pet_trainval_dataframe,
)


DEFAULT_CONFIG_PATH = Path("/workspace/config/classification/siglip2_oxford_pet_zeroshot.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot SigLIP 2 classification experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def prettify_label(label: str) -> str:
    return label.replace("_", " ")


def build_raw_image_collate_fn():
    def collate_fn(samples: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": [sample["image"] for sample in samples],
            "label": torch.tensor([int(sample["label"]) for sample in samples], dtype=torch.long),
            "image_path": [str(sample["image_path"]) for sample in samples],
            "image_stem": [str(sample["image_stem"]) for sample in samples],
            "breed_name": [str(sample["breed_name"]) for sample in samples],
        }

    return collate_fn


@torch.no_grad()
def predict_split(
    *,
    model,
    processor,
    dataloader: DataLoader,
    candidate_texts: list[str],
    device: str,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []

    for batch in dataloader:
        encoded = processor(
            text=candidate_texts,
            images=batch["image"],
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        logits = outputs.logits_per_image
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        for index in range(len(batch["image_path"])):
            pred_idx = int(predictions[index].item())
            rows.append(
                {
                    "image_path": batch["image_path"][index],
                    "image_stem": batch["image_stem"][index],
                    "breed_name": batch["breed_name"][index],
                    "label": int(batch["label"][index].item()),
                    "prediction": pred_idx,
                    "confidence": float(probabilities[index, pred_idx].item()),
                }
            )

    return pd.DataFrame(rows)


def evaluate_and_save(
    *,
    model,
    processor,
    dataframe: pd.DataFrame,
    candidate_texts: list[str],
    batch_size: int,
    num_workers: int,
    device: str,
    output_dir: Path,
    save_detailed_outputs: bool,
) -> tuple[float, pd.DataFrame]:
    dataset = OxfordIIITPetClassificationDataset(dataframe, transform=None)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=build_raw_image_collate_fn(),
    )
    prediction_df = predict_split(
        model=model,
        processor=processor,
        dataloader=dataloader,
        candidate_texts=candidate_texts,
        device=device,
    )
    accuracy = float((prediction_df["label"] == prediction_df["prediction"]).mean())

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"num_examples": int(len(prediction_df)), "accuracy": accuracy}
    save_json(output_dir / "summary.json", summary)

    if save_detailed_outputs:
        prediction_df.to_csv(output_dir / "predictions.csv", index=False)
        build_classification_report(
            prediction_df,
            target_names=[text.removeprefix("a photo of a ") for text in candidate_texts],
        ).to_csv(output_dir / "classification_report.csv")
        build_confusion_matrix(
            prediction_df,
            num_classes=len(candidate_texts),
        ).to_csv(output_dir / "confusion_matrix.csv", index=False)

    return accuracy, prediction_df


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())

    output_dir = Path(raw["output_dir"])
    run_dir = output_dir / "zeroshot"
    valid_dir = output_dir / "valid_results"
    test_dir = output_dir / "best_test_results"

    full_df = load_oxford_pet_trainval_dataframe(raw["dataset_root"])
    _, valid_df = create_train_valid_split(
        full_df,
        valid_size=raw["split"]["valid_size"],
        random_state=raw["split"]["random_state"],
    )
    test_df = load_oxford_pet_test_dataframe(raw["dataset_root"])
    label_mapping = build_label_mapping(full_df)
    candidate_labels = [prettify_label(label_mapping[index + 1]) for index in range(len(label_mapping))]
    prompt_template = raw.get("prompt_template", "a photo of a {label}")
    candidate_texts = [prompt_template.format(label=label) for label in candidate_labels]

    processor = AutoProcessor.from_pretrained(raw["model_name"])
    model = AutoModel.from_pretrained(raw["model_name"]).to(args.device)

    batch_size = raw["eval"]["batch_size"]
    num_workers = raw["eval"]["num_workers"]

    valid_accuracy, _ = evaluate_and_save(
        model=model,
        processor=processor,
        dataframe=valid_df,
        candidate_texts=candidate_texts,
        batch_size=batch_size,
        num_workers=num_workers,
        device=args.device,
        output_dir=valid_dir,
        save_detailed_outputs=False,
    )
    test_accuracy, _ = evaluate_and_save(
        model=model,
        processor=processor,
        dataframe=test_df,
        candidate_texts=candidate_texts,
        batch_size=batch_size,
        num_workers=num_workers,
        device=args.device,
        output_dir=test_dir,
        save_detailed_outputs=True,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", raw)
    save_json(run_dir / "label_mapping.json", {str(key): value for key, value in label_mapping.items()})

    experiment_summary = {
        "experiment_name": raw["experiment_name"],
        "mode": "zeroshot",
        "best_source": "zeroshot",
        "best_run_name": "zeroshot",
        "best_checkpoint_path": str(run_dir / "best.pt"),
        "best_valid_accuracy": valid_accuracy,
        "test_accuracy": test_accuracy,
        "model_name": raw["model_name"],
    }
    save_json(output_dir / "experiment_summary.json", experiment_summary)

    print("")
    print("zero-shot evaluation completed.")
    print(experiment_summary)


if __name__ == "__main__":
    main()
