"""Strict ZIP dataset validation and installation."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import random
import re
from zipfile import BadZipFile, ZipFile

import pandas as pd
from PIL import Image, UnidentifiedImageError


REQUIRED_COLUMNS = ("image_path", "scientific_name", "specimen_id")
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCIENTIFIC_NAME = re.compile(r"^[A-Z][A-Za-z-]+\s+[a-z][A-Za-z.-]+(?:\s+[a-z][A-Za-z.-]+)?$")


@dataclass
class DatasetInspection:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: pd.DataFrame | None = None
    fingerprint: str = ""

    @property
    def image_count(self) -> int:
        return 0 if self.manifest is None else len(self.manifest)

    @property
    def specimen_count(self) -> int:
        if self.manifest is None or "specimen_id" not in self.manifest:
            return 0
        return self.manifest["specimen_id"].nunique()

    @property
    def species_count(self) -> int:
        if self.manifest is None or "scientific_name" not in self.manifest:
            return 0
        return self.manifest["scientific_name"].nunique()


@dataclass(frozen=True)
class InstalledDataset:
    name: str
    path: Path
    manifest_path: Path
    fingerprint: str


def _safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def inspect_dataset_zip(blob: bytes) -> DatasetInspection:
    """Validate content, taxonomy labels, and images without extracting the ZIP."""
    errors: list[str] = []
    warnings: list[str] = []
    frame: pd.DataFrame | None = None

    try:
        archive = ZipFile(BytesIO(blob))
    except BadZipFile:
        return DatasetInspection(False, ["The uploaded file is not a readable ZIP archive."])

    with archive:
        file_names = [info.filename for info in archive.infolist() if not info.is_dir()]
        unsafe = [name for name in file_names if not _safe_archive_name(name)]
        if unsafe:
            errors.append("The ZIP contains unsafe or Windows-style paths: " + ", ".join(unsafe[:3]))

        if "manifest.csv" not in file_names:
            errors.append("manifest.csv must be located at the root of the ZIP.")

        unsupported = [
            name
            for name in file_names
            if name != "manifest.csv" and PurePosixPath(name).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES
        ]
        if unsupported:
            errors.append("Unsupported files are present: " + ", ".join(unsupported[:5]))

        images_outside_folder = [
            name
            for name in file_names
            if name != "manifest.csv" and not name.startswith("images/")
        ]
        if images_outside_folder:
            errors.append("Every image must be stored inside the images/ folder.")

        if "manifest.csv" in file_names:
            try:
                frame = pd.read_csv(BytesIO(archive.read("manifest.csv")), dtype=str).fillna("")
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                errors.append(f"manifest.csv is not a valid UTF-8 CSV: {exc}")

        if frame is not None:
            missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
            if missing:
                errors.append("manifest.csv is missing required columns: " + ", ".join(missing))
            else:
                frame = frame.copy()
                for column in REQUIRED_COLUMNS:
                    frame[column] = frame[column].astype(str).str.strip()

                if frame.empty:
                    errors.append("manifest.csv contains no image rows.")
                for column in REQUIRED_COLUMNS:
                    if (frame[column] == "").any():
                        errors.append(f"Column {column} contains empty values.")

                duplicated = frame[frame["image_path"].duplicated()]["image_path"].tolist()
                if duplicated:
                    errors.append("Duplicate image_path values: " + ", ".join(duplicated[:5]))

                invalid_paths = [
                    value
                    for value in frame["image_path"]
                    if not _safe_archive_name(value)
                    or not value.startswith("images/")
                    or PurePosixPath(value).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES
                ]
                if invalid_paths:
                    errors.append("Invalid image_path values: " + ", ".join(invalid_paths[:5]))

                archive_names = set(file_names)
                missing_images = [value for value in frame["image_path"] if value not in archive_names]
                if missing_images:
                    errors.append("Images referenced by the manifest are missing: " + ", ".join(missing_images[:5]))

                invalid_names = [
                    value for value in frame["scientific_name"] if not SCIENTIFIC_NAME.fullmatch(value)
                ]
                if invalid_names:
                    errors.append("Invalid scientific names: " + ", ".join(sorted(set(invalid_names))[:5]))

                conflicts = frame.groupby("specimen_id")["scientific_name"].nunique()
                conflicting_ids = conflicts[conflicts > 1].index.tolist()
                if conflicting_ids:
                    errors.append(
                        "A specimen_id cannot belong to multiple species: "
                        + ", ".join(conflicting_ids[:5])
                    )

                species_count = frame["scientific_name"].nunique()
                if species_count < 5:
                    errors.append("At least 5 species are required to produce Top-5 predictions.")

                specimens_per_species = (
                    frame.groupby("scientific_name")["specimen_id"].nunique().sort_values()
                )
                too_small = specimens_per_species[specimens_per_species < 3]
                if not too_small.empty:
                    details = ", ".join(f"{name} ({count})" for name, count in too_small.items())
                    errors.append("Each species needs at least 3 distinct specimens: " + details)

                low_sample = specimens_per_species[
                    (specimens_per_species >= 3) & (specimens_per_species < 20)
                ]
                if not low_sample.empty:
                    warnings.append(
                        f"{len(low_sample)} species have fewer than 20 specimens; accuracy may be unstable."
                    )

                for image_path in frame["image_path"]:
                    if image_path not in archive_names:
                        continue
                    try:
                        with Image.open(BytesIO(archive.read(image_path))) as image:
                            image.verify()
                    except (UnidentifiedImageError, OSError):
                        errors.append(f"Unreadable image: {image_path}")
                        if len(errors) >= 25:
                            break

    return DatasetInspection(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        manifest=frame,
        fingerprint=sha256(blob).hexdigest(),
    )


def specimen_safe_split(frame: pd.DataFrame, validation_fraction: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Assign every biological specimen to one split, stratified by species."""
    result = frame.copy()
    result["split"] = "train"
    rng = random.Random(seed)
    for species, rows in result.groupby("scientific_name"):
        specimen_ids = sorted(rows["specimen_id"].unique())
        rng.shuffle(specimen_ids)
        validation_count = max(1, round(len(specimen_ids) * validation_fraction))
        validation_ids = set(specimen_ids[:validation_count])
        mask = (result["scientific_name"] == species) & result["specimen_id"].isin(validation_ids)
        result.loc[mask, "split"] = "validation"
    return result


def install_dataset(blob: bytes, dataset_name: str, datasets_root: Path) -> InstalledDataset:
    """Install only manifest-referenced images after a successful validation."""
    inspection = inspect_dataset_zip(blob)
    if not inspection.valid or inspection.manifest is None:
        raise ValueError("The dataset package failed validation: " + "; ".join(inspection.errors))

    slug = re.sub(r"[^a-z0-9]+", "-", dataset_name.lower()).strip("-") or "dataset"
    destination = datasets_root / f"{slug}-{inspection.fingerprint[:10]}"
    destination.mkdir(parents=True, exist_ok=True)

    with ZipFile(BytesIO(blob)) as archive:
        for image_path in inspection.manifest["image_path"]:
            target = destination / PurePosixPath(image_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(image_path))

    split_frame = specimen_safe_split(inspection.manifest)
    manifest_path = destination / "manifest.csv"
    split_frame.to_csv(manifest_path, index=False)
    (destination / "dataset.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "fingerprint": inspection.fingerprint,
                "images": len(split_frame),
                "specimens": int(split_frame["specimen_id"].nunique()),
                "species": int(split_frame["scientific_name"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return InstalledDataset(dataset_name, destination, manifest_path, inspection.fingerprint)
