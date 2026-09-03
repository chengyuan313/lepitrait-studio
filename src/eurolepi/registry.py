"""Filesystem model registry shared by all three GUI workflows."""

from __future__ import annotations

import json
from pathlib import Path

from eurolepi.types import ModelRecord


class ModelRegistry:
    def __init__(self, models_root: Path):
        self.models_root = models_root

    def list_models(self) -> list[ModelRecord]:
        if not self.models_root.exists():
            return []
        records: list[ModelRecord] = []
        for metadata_path in self.models_root.glob("*/model.json"):
            try:
                records.append(self._read(metadata_path))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def get(self, model_id: str) -> ModelRecord:
        metadata_path = self.models_root / model_id / "model.json"
        if not metadata_path.is_file():
            raise KeyError(f"Model not found: {model_id}")
        return self._read(metadata_path)

    @staticmethod
    def _read(metadata_path: Path) -> ModelRecord:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_path = metadata_path.parent
        if not (model_path / "best.pt").is_file():
            raise ValueError("The model package has no best.pt checkpoint.")
        return ModelRecord(
            model_id=payload["model_id"],
            dataset_name=payload["dataset_name"],
            created_at=payload["created_at"],
            path=model_path,
            backbone=payload["backbone"],
            species=tuple(payload["species"]),
            references=dict(payload.get("references", {})),
            metadata=payload,
        )


def write_model_metadata(model_path: Path, payload: dict) -> Path:
    metadata_path = model_path / "model.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path

