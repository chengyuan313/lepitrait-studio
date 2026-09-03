"""Transparent segmentation baseline for standardized light backgrounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    foreground_fraction: float
    component_count: int


def segment_specimen(
    rgb: np.ndarray,
    background_threshold: int = 238,
    min_saturation: int = 10,
    keep_largest: bool = True,
) -> SegmentationResult:
    """Segment a centered specimen photographed on a pale neutral background.

    This deliberately simple baseline makes failure visible. It must not be used
    as a silent substitute for a validated LEPY segmentation model.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    dark = image.min(axis=2) < background_threshold
    chromatic = (image.max(axis=2) - image.min(axis=2)) > min_saturation
    mask = dark | chromatic
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)

    labels, count = ndimage.label(mask)
    if keep_largest and count:
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        mask = labels == int(np.argmax(areas))
    return SegmentationResult(
        mask=mask.astype(bool),
        foreground_fraction=float(mask.mean()),
        component_count=int(count),
    )

