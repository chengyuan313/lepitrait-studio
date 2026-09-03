"""Small dependency-free domain objects shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ImageDomain(str, Enum):
    MUSEUM_STANDARDIZED = "museum_standardized"
    FIELD_STANDARDIZED = "field_standardized"
    FIELD_IN_SITU = "field_in_situ"


class ViewSide(str, Enum):
    DORSAL = "dorsal"
    VENTRAL = "ventral"
    LATERAL = "lateral"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Prediction:
    scientific_name: str
    probability: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdentificationResult:
    predictions: tuple[Prediction, ...]
    rejected: bool
    threshold: float
    model_name: str

    @property
    def decision(self) -> str:
        if self.rejected or not self.predictions:
            return "unknown_or_review"
        return self.predictions[0].scientific_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "rejected": self.rejected,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "predictions": [item.to_dict() for item in self.predictions],
        }

