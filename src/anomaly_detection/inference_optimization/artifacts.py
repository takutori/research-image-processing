from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelArtifact:
    model_name: str
    summary_path: Path
    summary: dict[str, Any]
    best_artifact_path: Path
    best_model_config: dict[str, Any]


def load_model_artifact(model_name: str, summary_path: Path) -> ModelArtifact:
    if not summary_path.exists():
        raise FileNotFoundError(f"experiment_summary.json not found for {model_name}: {summary_path}")
    summary = json.loads(summary_path.read_text())
    artifact_value = summary.get("best_artifact_path")
    if not artifact_value:
        raise KeyError(f"best_artifact_path is missing in {summary_path}")

    return ModelArtifact(
        model_name=model_name,
        summary_path=summary_path,
        summary=summary,
        best_artifact_path=Path(artifact_value),
        best_model_config=dict(summary.get("best_model_config") or {}),
    )


def find_category_checkpoint(artifact_path: Path, category: str) -> Path:
    search_dir = artifact_path / category / "anomalib_run"
    ckpts = sorted(search_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {search_dir}")
    return ckpts[-1]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True))
