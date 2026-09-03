"""Reviewable label OCR and conservative Darwin Core-oriented parsing."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .schema import SpecimenMetadata


SCIENTIFIC_NAME = re.compile(
    r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{2,})(?:\s+([a-z][a-z-]{2,}))?\b"
)
UPPER_SCIENTIFIC_NAME = re.compile(r"\b([A-Z]{4,})\s+([A-Z]{4,})\.\B")
ISO_DATE = re.compile(r"\b(18|19|20)\d{2}[-/.](0?[1-9]|1[0-2])[-/.]([0-2]?\d|3[01])\b")
DMY_DATE = re.compile(
    r"\b([0-2]?\d|3[01])\s*[-/.]\s*(0?\d|1[0-2])\s*[-/.]\s*((?:18|19|20)\d{2})\b"
)
ROMAN_DATE = re.compile(
    r"\b([0-2]?\d|3[01])\s*[./-]?\s*"
    r"(i{1,3}|iv|v|vi{0,3}|ix|x|xi|xii)\s*[./-]?\s*((?:18|19|20)\d{2})\b",
    re.IGNORECASE,
)
TEXT_DATE = re.compile(
    r"\b([0-2]?\d|3[01])(?:st|nd|rd|th)?\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[,.]?\s+((?:18|19|20)\d{2})\b",
    re.IGNORECASE,
)
COORDINATES = re.compile(r"(-?\d{1,2}(?:\.\d+)?)\s*[,;/]\s*(-?\d{1,3}(?:\.\d+)?)")
CATALOG_NUMBER = re.compile(
    r"\b(?:NHMUK\s*[0-9O][0-9O\s]{6,}[0-9O]|BMNH\s*\(\s*E\s*\)\s*\d{6,})\b",
    re.IGNORECASE,
)
NON_TAXON_WORDS = {
    "bequest",
    "bright",
    "collection",
    "coll",
    "forest",
    "holotype",
    "museum",
    "subprov",
    "type",
}

ROMAN_MONTHS = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "xi": 11,
    "xii": 12,
}
TEXT_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class OcrUnavailableError(RuntimeError):
    """Raised when no local Tesseract executable can be found."""


@dataclass(frozen=True)
class LabelCrop:
    """Pixel bounds for a label panel, using PIL's left/top/right/bottom convention."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True)
class OcrResult:
    text: str
    engine: str
    crop: LabelCrop | None = None


def suggested_label_crop(
    image: Image.Image,
    *,
    left_fraction: float = 0.56,
    top_fraction: float = 0.0,
    right_fraction: float = 1.0,
    bottom_fraction: float = 0.82,
) -> LabelCrop:
    """Return a reviewable default crop for the project's label-on-right capture layout.

    This is deliberately a layout suggestion, not an ML detection claim. The GUI exposes all
    four bounds because museum drawers and historical imaging batches do vary.
    """

    width, height = image.size
    fractions = (left_fraction, top_fraction, right_fraction, bottom_fraction)
    if any(not 0 <= value <= 1 for value in fractions):
        raise ValueError("Crop fractions must be between 0 and 1.")
    if left_fraction >= right_fraction or top_fraction >= bottom_fraction:
        raise ValueError("Crop bounds must have positive width and height.")
    return LabelCrop(
        left=round(width * left_fraction),
        top=round(height * top_fraction),
        right=round(width * right_fraction),
        bottom=round(height * bottom_fraction),
    )


def prepare_label_image(image: Image.Image, *, minimum_width: int = 1200) -> Image.Image:
    """Normalize a label panel for small, mixed printed/handwritten museum text."""

    prepared = ImageOps.exif_transpose(image).convert("L")
    if prepared.width < minimum_width:
        scale = minimum_width / prepared.width
        prepared = prepared.resize(
            (minimum_width, max(1, round(prepared.height * scale))), Image.Resampling.LANCZOS
        )
    prepared = ImageOps.autocontrast(prepared, cutoff=0.5)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.25)
    return prepared.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))


def _find_tesseract(tesseract_cmd: str | None = None) -> str:
    configured = tesseract_cmd or os.environ.get("TESSERACT_CMD")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        resolved = shutil.which(configured)
        if resolved:
            return resolved
    resolved = shutil.which("tesseract")
    if resolved:
        return resolved
    raise OcrUnavailableError(
        "Tesseract was not found. Install Tesseract OCR and restart the app, or set "
        "TESSERACT_CMD to the full path of tesseract.exe."
    )


def _clean_ocr_text(text: str) -> str:
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" |_")
        if not line:
            continue
        compact = re.sub(r"[^A-Za-z0-9]", "", line)
        if len(compact) >= 14 and len(set(compact.lower())) <= 4:
            # Ruler ticks and borders often become long, repetitive OCR artefacts.
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def ocr_label(
    image: Image.Image,
    *,
    tesseract_cmd: str | None = None,
    language: str = "eng",
    page_segmentation_mode: int = 11,
    crop: LabelCrop | None = None,
) -> OcrResult:
    """Run offline Tesseract OCR on a label image and return editable verbatim text."""

    command = _find_tesseract(tesseract_cmd)
    prepared = prepare_label_image(image)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temporary_path = handle.name
        prepared.save(temporary_path)
        completed = subprocess.run(
            [command, temporary_path, "stdout", "-l", language, "--psm", str(page_segmentation_mode)],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("OCR timed out after 90 seconds.") from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)

    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"Tesseract OCR failed: {detail}")
    return OcrResult(
        text=_clean_ocr_text(completed.stdout),
        engine="tesseract-5/eng",
        crop=crop,
    )


def parse_label(text: str) -> SpecimenMetadata:
    metadata = SpecimenMetadata(verbatim_label=text or None)
    normalized = re.sub(r"[ \t]+", " ", text)

    for line in normalized.splitlines() or [normalized]:
        name = SCIENTIFIC_NAME.search(line)
        if name and not ({part.lower() for part in name.groups() if part} & NON_TAXON_WORDS):
            metadata.verbatim_scientific_name = " ".join(part for part in name.groups() if part)
            metadata.field_confidence["verbatim_scientific_name"] = 0.55
            break
        upper_name = UPPER_SCIENTIFIC_NAME.search(line)
        if upper_name and not ({part.lower() for part in upper_name.groups()} & NON_TAXON_WORDS):
            metadata.verbatim_scientific_name = (
                f"{upper_name.group(1).title()} {upper_name.group(2).lower()}"
            )
            metadata.field_confidence["verbatim_scientific_name"] = 0.4
            break

    _parse_date(normalized, metadata)

    coords = COORDINATES.search(normalized)
    if coords:
        lat, lon = float(coords.group(1)), float(coords.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            metadata.decimal_latitude = lat
            metadata.decimal_longitude = lon
            metadata.field_confidence["coordinates"] = 0.7

    catalog = CATALOG_NUMBER.search(normalized)
    if catalog:
        value = re.sub(r"\s+", "", catalog.group(0).upper())
        if value.startswith("BMNH"):
            value = re.sub(r"BMNH\(E\)", "BMNH(E) ", value)
            metadata.institution_code = "BMNH"
        else:
            value = "NHMUK" + value.removeprefix("NHMUK").replace("O", "0")
            metadata.institution_code = "NHMUK"
        metadata.catalog_number = value
        metadata.field_confidence["catalog_number"] = 0.9

    if re.search(r"\bHOLO\s*-?\s*TYPE\b", normalized, re.IGNORECASE):
        metadata.type_status = "holotype"
        metadata.field_confidence["type_status"] = 0.9
    elif re.search(r"\bTYPE\b", normalized, re.IGNORECASE):
        metadata.type_status = "type"
        metadata.field_confidence["type_status"] = 0.75

    if "♀" in normalized:
        metadata.sex = "female"
        metadata.field_confidence["sex"] = 0.95
    elif "♂" in normalized:
        metadata.sex = "male"
        metadata.field_confidence["sex"] = 0.95
    return metadata


def _parse_date(text: str, metadata: SpecimenMetadata) -> None:
    iso = ISO_DATE.search(text)
    if iso:
        raw = iso.group(0)
        parts = re.split(r"[-/.]", raw)
        _set_date(metadata, raw, int(parts[0]), int(parts[1]), int(parts[2]), 0.8)
        return

    dmy = DMY_DATE.search(text)
    if dmy:
        _set_date(metadata, dmy.group(0), int(dmy.group(3)), int(dmy.group(2)), int(dmy.group(1)), 0.72)
        return

    roman = ROMAN_DATE.search(text)
    if roman:
        _set_date(
            metadata,
            roman.group(0),
            int(roman.group(3)),
            ROMAN_MONTHS[roman.group(2).lower()],
            int(roman.group(1)),
            0.72,
        )
        return

    textual = TEXT_DATE.search(text)
    if textual:
        _set_date(
            metadata,
            textual.group(0),
            int(textual.group(3)),
            TEXT_MONTHS[textual.group(2)[:3].lower()],
            int(textual.group(1)),
            0.72,
        )


def _set_date(
    metadata: SpecimenMetadata,
    raw: str,
    year: int,
    month: int,
    day: int,
    confidence: float,
) -> None:
    metadata.verbatim_event_date = raw
    try:
        metadata.event_date = date(year, month, day)
        metadata.field_confidence["event_date"] = confidence
    except ValueError:
        pass
