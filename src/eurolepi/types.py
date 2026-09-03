"""Shared immutable records used by the GUI, trainer, and predictors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    rank: int
    scientific_name: str
    probability: float
    reference_image: Path | None = None


@dataclass(frozen=True)
class IdentificationResult:
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    dataset_name: str
    created_at: str
    path: Path
    backbone: str
    species: tuple[str, ...]
    references: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"{self.dataset_name} · {len(self.species)} species · {self.created_at[:10]}"

