from __future__ import annotations

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix


def _extract_logits(model_output):
    return model_output.logits if hasattr(model_output, "logits") else model_output


def _prepare_model_inputs(batch: dict[str, object], device: str):
    if "model_inputs" in batch:
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch["model_inputs"].items()
        }
    return batch["image"].to(device)


def _forward_model(model, batch: dict[str, object], device: str):
    model_inputs = _prepare_model_inputs(batch, device)
    if isinstance(model_inputs, dict):
        return model(**model_inputs)
    return model(model_inputs)


@torch.no_grad()
def collect_predictions(model, dataloader, device: str = "cuda") -> pd.DataFrame:
    model.eval()
    records: list[dict[str, object]] = []

    for batch in dataloader:
        labels = batch["label"].to(device)
        logits = _extract_logits(_forward_model(model, batch, device))
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        for index in range(labels.size(0)):
            records.append(
                {
                    "image_path": batch["image_path"][index],
                    "image_stem": batch["image_stem"][index],
                    "breed_name": batch["breed_name"][index],
                    "label": int(labels[index].item()),
                    "prediction": int(predictions[index].item()),
                    "confidence": float(probabilities[index, predictions[index]].item()),
                }
            )

    return pd.DataFrame(records)


def build_classification_report(prediction_df: pd.DataFrame, target_names: list[str] | None = None) -> pd.DataFrame:
    labels = list(range(len(target_names))) if target_names is not None else None
    report = classification_report(
        prediction_df["label"],
        prediction_df["prediction"],
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def build_confusion_matrix(prediction_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    matrix = confusion_matrix(
        prediction_df["label"],
        prediction_df["prediction"],
        labels=list(range(num_classes)),
    )
    return pd.DataFrame(matrix)
