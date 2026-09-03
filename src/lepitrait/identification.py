"""Species-identification contracts that prevent label leakage."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .schema import TaxonPrediction


class SpeciesIdentifier(ABC):
    name: str
    version: str

    @abstractmethod
    def predict(self, specimen_rgb: np.ndarray, specimen_mask: np.ndarray, top_k: int = 5) -> list[TaxonPrediction]:
        """Predict from a specimen-only image. Labels and rulers must already be removed."""


class UnconfiguredIdentifier(SpeciesIdentifier):
    name = "not-configured"
    version = "0"

    def predict(self, specimen_rgb: np.ndarray, specimen_mask: np.ndarray, top_k: int = 5) -> list[TaxonPrediction]:
        return []


def masked_crop(rgb: np.ndarray, mask: np.ndarray, padding: int = 12) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Cannot crop an empty specimen mask")
    y0, x0 = np.maximum(coords.min(axis=0) - padding, 0)
    y1, x1 = np.minimum(coords.max(axis=0) + padding + 1, mask.shape)
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 235
    return crop

