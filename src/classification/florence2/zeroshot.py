from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.common import (
    build_classification_report,
    build_confusion_matrix,
    build_label_mapping,
    create_train_valid_split,
    limit_dataframe_examples,
    load_oxford_pet_test_dataframe,
    load_oxford_pet_trainval_dataframe,
)


DEFAULT_CONFIG_PATH = Path("/workspace/config/classification/florence2_oxford_pet_zeroshot.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot Florence-2 classification experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def prettify_label(label: str) -> str:
    return label.replace("_", " ")


def canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def map_prediction_to_label(prediction_text: str, candidate_labels: list[str]) -> int:
    canonical_candidates = [canonicalize(label) for label in candidate_labels]
    canonical_prediction = canonicalize(prediction_text)
    if canonical_prediction in canonical_candidates:
        return canonical_candidates.index(canonical_prediction)

    for index, candidate in enumerate(canonical_candidates):
        if candidate in canonical_prediction or canonical_prediction in candidate:
            return index

    match = difflib.get_close_matches(canonical_prediction, canonical_candidates, n=1, cutoff=0.0)
    if match:
        return canonical_candidates.index(match[0])
    return 0


@torch.no_grad()
def evaluate_dataframe(
    *,
    model,
    processor,
    dataframe: pd.DataFrame,
    prompt: str,
    candidate_labels: list[str],
    device: str,
    max_new_tokens: int,
    output_dir: Path,
    save_detailed_outputs: bool,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for row in tqdm(list(dataframe.itertuples(index=False)), desc="florence2-zero-shot", leave=False):
        image = Image.open(row.image_path).convert("RGB")
        encoded = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            use_cache=False,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        prediction_index = map_prediction_to_label(generated_text, candidate_labels)
        rows.append(
            {
                "image_path": str(row.image_path),
                "image_stem": row.image_stem,
                "breed_name": row.breed_name,
                "label": int(row.class_id) - 1,
                "prediction": prediction_index,
                "confidence": 1.0,
                "raw_prediction": generated_text,
            }
        )

    prediction_df = pd.DataFrame(rows)
    accuracy = float((prediction_df["label"] == prediction_df["prediction"]).mean())
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"num_examples": int(len(prediction_df)), "accuracy": accuracy}
    save_json(output_dir / "summary.json", summary)

    if save_detailed_outputs:
        prediction_df.to_csv(output_dir / "predictions.csv", index=False)
        build_classification_report(prediction_df, target_names=candidate_labels).to_csv(output_dir / "classification_report.csv")
        build_confusion_matrix(prediction_df, num_classes=len(candidate_labels)).to_csv(output_dir / "confusion_matrix.csv", index=False)

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
    subset = raw.get("subset", {})
    valid_df = limit_dataframe_examples(valid_df, subset.get("valid_examples"), random_state=raw["split"]["random_state"])
    test_df = limit_dataframe_examples(test_df, subset.get("test_examples"), random_state=raw["split"]["random_state"])
    label_mapping = build_label_mapping(full_df)
    candidate_labels = [prettify_label(label_mapping[index + 1]) for index in range(len(label_mapping))]
    prompt = raw.get(
        "prompt",
        "What breed of pet is shown in the image? Choose exactly one from: {labels}",
    ).format(labels=", ".join(candidate_labels))

    processor = AutoProcessor.from_pretrained(raw["model_name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        raw["model_name"],
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device)

    valid_accuracy, _ = evaluate_dataframe(
        model=model,
        processor=processor,
        dataframe=valid_df,
        prompt=prompt,
        candidate_labels=candidate_labels,
        device=args.device,
        max_new_tokens=raw["eval"]["max_new_tokens"],
        output_dir=valid_dir,
        save_detailed_outputs=False,
    )
    test_accuracy, _ = evaluate_dataframe(
        model=model,
        processor=processor,
        dataframe=test_df,
        prompt=prompt,
        candidate_labels=candidate_labels,
        device=args.device,
        max_new_tokens=raw["eval"]["max_new_tokens"],
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
