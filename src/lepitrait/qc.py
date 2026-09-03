"""Image and segmentation quality checks."""

from __future__ import annotations

import numpy as np

from .schema import QualityFlag, Severity


def image_quality_flags(rgb: np.ndarray, mask: np.ndarray) -> list[QualityFlag]:
    flags: list[QualityFlag] = []
    height, width = rgb.shape[:2]
    gray = rgb.astype(np.float32).mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness = float((gx.var() + gy.var()) / 2)
    clipped_high = float(np.mean(rgb >= 250))
    clipped_low = float(np.mean(rgb <= 5))
    fraction = float(mask.mean())

    if min(height, width) < 1200:
        flags.append(QualityFlag(code="LOW_RESOLUTION", severity=Severity.WARNING, message="Shortest image edge is below 1200 px.", value=min(height, width)))
    if sharpness < 35:
        flags.append(QualityFlag(code="POSSIBLE_BLUR", severity=Severity.WARNING, message="Edge variance is low; inspect focus manually.", value=round(sharpness, 2)))
    if clipped_high > 0.40:
        flags.append(QualityFlag(code="HIGH_HIGHLIGHT_CLIPPING", severity=Severity.WARNING, message="A large fraction of pixels is near white.", value=round(clipped_high, 4)))
    if clipped_low > 0.08:
        flags.append(QualityFlag(code="HIGH_SHADOW_CLIPPING", severity=Severity.WARNING, message="A large fraction of pixels is near black.", value=round(clipped_low, 4)))
    if fraction < 0.03:
        flags.append(QualityFlag(code="MASK_TOO_SMALL", severity=Severity.ERROR, message="Detected specimen occupies less than 3% of the image.", value=round(fraction, 4)))
    if fraction > 0.75:
        flags.append(QualityFlag(code="MASK_TOO_LARGE", severity=Severity.ERROR, message="Detected specimen occupies more than 75% of the image.", value=round(fraction, 4)))
    if not flags:
        flags.append(QualityFlag(code="BASELINE_QC_PASS", severity=Severity.INFO, message="No baseline image-quality warning was triggered."))
    return flags

