"""Conservative label OCR hooks and Darwin Core-oriented field parsing."""

from __future__ import annotations

import re
from datetime import date

from PIL import Image

from .schema import SpecimenMetadata


SCIENTIFIC_NAME = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{2,})(?:\s+([a-z][a-z-]{2,}))?\b")
ISO_DATE = re.compile(r"\b(18|19|20)\d{2}[-/.](0?[1-9]|1[0-2])[-/.]([0-2]?\d|3[01])\b")
COORDINATES = re.compile(r"(-?\d{1,2}(?:\.\d+)?)\s*[,;/]\s*(-?\d{1,3}(?:\.\d+)?)")


def ocr_label(image: Image.Image) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("OCR is not installed. Install with: pip install -e '.[ocr]'") from exc
    return pytesseract.image_to_string(image, config="--psm 6").strip()


def parse_label(text: str) -> SpecimenMetadata:
    metadata = SpecimenMetadata(verbatim_label=text or None)
    name = SCIENTIFIC_NAME.search(text)
    if name:
        metadata.verbatim_scientific_name = " ".join(part for part in name.groups() if part)
        metadata.field_confidence["verbatim_scientific_name"] = 0.55

    found_date = ISO_DATE.search(text)
    if found_date:
        raw = found_date.group(0)
        metadata.verbatim_event_date = raw
        parts = re.split(r"[-/.]", raw)
        try:
            metadata.event_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            metadata.field_confidence["event_date"] = 0.8
        except ValueError:
            pass

    coords = COORDINATES.search(text)
    if coords:
        lat, lon = float(coords.group(1)), float(coords.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            metadata.decimal_latitude = lat
            metadata.decimal_longitude = lon
            metadata.field_confidence["coordinates"] = 0.7
    return metadata

