"""Scale-aware morphology and calibrated-image colour summaries."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .image import srgb_to_lab
from .schema import ColourSummary, TraitMeasurement


def morphology_measurements(mask: np.ndarray, pixels_per_mm: float | None) -> list[TraitMeasurement]:
    coords = np.argwhere(mask)
    if len(coords) < 3:
        return []

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    area_px = float(mask.sum())
    span_x_px = float(x1 - x0 + 1)
    span_y_px = float(y1 - y0 + 1)

    centered = coords[:, ::-1] - coords[:, ::-1].mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(centered, rowvar=False))
    eigenvalues = np.maximum(eigenvalues, 0)
    major_px = float(4 * math.sqrt(eigenvalues[-1]))
    minor_px = float(4 * math.sqrt(eigenvalues[0]))
    eroded = ndimage.binary_erosion(mask)
    perimeter_px = float(np.count_nonzero(mask ^ eroded))

    raw = {
        "silhouette_area": (area_px, "px2"),
        "horizontal_span": (span_x_px, "px"),
        "vertical_span": (span_y_px, "px"),
        "major_axis": (major_px, "px"),
        "minor_axis": (minor_px, "px"),
        "silhouette_perimeter": (perimeter_px, "px"),
    }
    measurements = [
        TraitMeasurement(name=name, value=value, unit=unit, method="baseline_mask_v1")
        for name, (value, unit) in raw.items()
    ]

    if pixels_per_mm:
        converted = {
            "silhouette_area": (area_px / pixels_per_mm**2, "mm2"),
            "horizontal_span": (span_x_px / pixels_per_mm, "mm"),
            "vertical_span": (span_y_px / pixels_per_mm, "mm"),
            "major_axis": (major_px / pixels_per_mm, "mm"),
            "minor_axis": (minor_px / pixels_per_mm, "mm"),
            "silhouette_perimeter": (perimeter_px / pixels_per_mm, "mm"),
        }
        measurements.extend(
            TraitMeasurement(name=name, value=value, unit=unit, method="baseline_mask_v1")
            for name, (value, unit) in converted.items()
        )
    return measurements


def colour_summary(rgb: np.ndarray, mask: np.ndarray, calibrated: bool) -> ColourSummary | None:
    if not np.any(mask):
        return None
    lab = srgb_to_lab(rgb)[mask]
    l_values, a_values, b_values = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.sqrt(a_values**2 + b_values**2)
    mean_a = float(a_values.mean())
    mean_b = float(b_values.mean())
    hue = float((np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360)
    return ColourSummary(
        l_mean=float(l_values.mean()),
        l_sd=float(l_values.std()),
        a_mean=mean_a,
        b_mean=mean_b,
        chroma_mean=float(chroma.mean()),
        hue_degrees=hue,
        dark_fraction=float(np.mean(l_values < 30)),
        calibrated=calibrated,
    )

