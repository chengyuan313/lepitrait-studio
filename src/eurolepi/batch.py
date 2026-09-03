"""Batch prediction table generation."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError


RESULT_COLUMNS = ["image_name"] + [
    item
    for rank in range(1, 6)
    for item in (f"top{rank}_species", f"top{rank}_probability")
] + ["error"]


def identify_batch(files: Iterable[tuple[str, bytes]], identifier) -> pd.DataFrame:
    """Identify image byte streams and always return one row per input file."""
    rows: list[dict[str, object]] = []
    for name, blob in files:
        row: dict[str, object] = {column: "" for column in RESULT_COLUMNS}
        row["image_name"] = name
        try:
            with Image.open(BytesIO(blob)) as source:
                image = source.convert("RGB")
            result = identifier.predict(image, top_k=5)
            for candidate in result.candidates:
                row[f"top{candidate.rank}_species"] = candidate.scientific_name
                row[f"top{candidate.rank}_probability"] = round(candidate.probability, 6)
        except (UnidentifiedImageError, OSError, RuntimeError, ValueError, KeyError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)

