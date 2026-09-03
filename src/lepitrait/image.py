"""Image loading, colour conversion and simple visualization helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(source: str | Path | bytes | BytesIO | Image.Image) -> np.ndarray:
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, bytes):
        image = Image.open(BytesIO(source))
    else:
        image = Image.open(source)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 sRGB to CIE L*a*b* using a D65 reference white."""
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    lab = np.empty_like(f)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    tint = np.array([31, 143, 135], dtype=np.float32)
    result[mask] = result[mask] * 0.68 + tint * 0.32
    boundary = mask ^ _binary_erode(mask)
    result[boundary] = np.array([224, 113, 54], dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=1)

