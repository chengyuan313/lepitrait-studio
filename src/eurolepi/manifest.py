"""Dataset manifest validation and specimen-safe train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import ImageDomain, ViewSide


REQUIRED_COLUMNS = {
    "image_id",
    "specimen_id",
    "image_path",
    "scientific_name",
    "genus",
    "family",
    "view",
    "domain",
    "dataset_source",
    "license",
    "label_pixels_removed",
}
ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass
class ManifestReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    images: int = 0
    specimens: int = 0
    species: int = 0

    @property
    def valid(self) -> bool:
        return not self.errors


def _as_bool(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def validate_manifest(
    frame: pd.DataFrame,
    *,
    require_files: bool = True,
    require_split: bool = True,
    minimum_train_images: int = 50,
) -> ManifestReport:
    report = ManifestReport(images=len(frame))
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        report.errors.append(f"Missing columns: {', '.join(sorted(missing))}")
        return report
    if require_split and "split" not in frame.columns:
        report.errors.append("Missing column: split")
        return report
    if frame.empty:
        report.errors.append("Manifest has no image records.")
        return report

    required = sorted(REQUIRED_COLUMNS | ({"split"} if require_split else set()))
    empty_columns = [
        column
        for column in required
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any()
    ]
    if empty_columns:
        report.errors.append(f"Required columns contain empty values: {', '.join(empty_columns)}")

    report.specimens = int(frame["specimen_id"].nunique())
    report.species = int(frame["scientific_name"].nunique())

    duplicate_ids = frame.loc[frame["image_id"].duplicated(), "image_id"].astype(str).tolist()
    if duplicate_ids:
        report.errors.append(f"Duplicate image_id values: {duplicate_ids[:10]}")
    duplicate_paths = frame.loc[frame["image_path"].duplicated(), "image_path"].astype(str).tolist()
    if duplicate_paths:
        report.errors.append(f"Duplicate image paths: {duplicate_paths[:10]}")

    allowed_domains = {item.value for item in ImageDomain}
    invalid_domains = sorted(set(frame["domain"].astype(str)) - allowed_domains)
    if invalid_domains:
        report.errors.append(f"Unknown domain values: {invalid_domains}")
    allowed_views = {item.value for item in ViewSide}
    invalid_views = sorted(set(frame["view"].astype(str)) - allowed_views)
    if invalid_views:
        report.errors.append(f"Unknown view values: {invalid_views}")

    label_flags = frame["label_pixels_removed"].map(_as_bool)
    invalid_flags = frame.loc[label_flags.isna(), "image_id"].astype(str).tolist()
    if invalid_flags:
        report.errors.append(f"Invalid label_pixels_removed values: {invalid_flags[:10]}")
    visible_labels = frame.loc[label_flags.eq(False), "image_id"].astype(str).tolist()
    if visible_labels:
        report.errors.append(
            "Classifier inputs still contain label pixels: " + ", ".join(visible_labels[:10])
        )

    specimen_labels = frame.groupby("specimen_id")["scientific_name"].nunique()
    conflicts = specimen_labels[specimen_labels > 1].index.astype(str).tolist()
    if conflicts:
        report.errors.append(f"Specimens with conflicting species labels: {conflicts[:10]}")

    genus_mismatch = frame[
        frame.apply(
            lambda row: str(row["scientific_name"]).split()[0] != str(row["genus"]).strip(),
            axis=1,
        )
    ]["image_id"].astype(str).tolist()
    if genus_mismatch:
        report.warnings.append(f"Scientific name/genus mismatch: {genus_mismatch[:10]}")

    if require_split:
        invalid_splits = sorted(set(frame["split"].astype(str)) - ALLOWED_SPLITS)
        if invalid_splits:
            report.errors.append(f"Unknown split values: {invalid_splits}")
        leakage = frame.groupby("specimen_id")["split"].nunique()
        leaking_ids = leakage[leakage > 1].index.astype(str).tolist()
        if leaking_ids:
            report.errors.append(f"Specimen leakage across splits: {leaking_ids[:10]}")

        train_species = set(frame.loc[frame["split"] == "train", "scientific_name"])
        evaluation_species = set(
            frame.loc[frame["split"].isin(["validation", "test"]), "scientific_name"]
        )
        missing_train = sorted(evaluation_species - train_species)
        if missing_train:
            report.errors.append(f"Evaluation species absent from training: {missing_train[:10]}")

        train_counts = frame.loc[frame["split"] == "train", "scientific_name"].value_counts()
        sparse = train_counts[train_counts < minimum_train_images]
        if not sparse.empty:
            preview = ", ".join(f"{name} ({count})" for name, count in sparse.head(10).items())
            report.warnings.append(
                f"Species below the recommended {minimum_train_images} training images: {preview}"
            )

    if require_files:
        missing_files = [str(path) for path in frame["image_path"] if not Path(path).is_file()]
        if missing_files:
            report.errors.append(f"Missing image files: {missing_files[:10]}")
    return report


def _partition_counts(size: int, validation_fraction: float, test_fraction: float) -> tuple[int, int]:
    if size < 3:
        return 0, 0
    test = max(1, int(round(size * test_fraction)))
    validation = max(1, int(round(size * validation_fraction)))
    while test + validation >= size:
        if test >= validation and test > 1:
            test -= 1
        elif validation > 1:
            validation -= 1
        else:
            break
    return validation, test


def assign_grouped_splits(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.DataFrame:
    """Split within species while keeping every specimen in exactly one partition."""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must both be positive.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one.")
    preflight = validate_manifest(frame, require_files=False, require_split=False)
    if not preflight.valid:
        raise ValueError("Invalid manifest: " + "; ".join(preflight.errors))

    rng = np.random.default_rng(seed)
    specimen_split: dict[str, str] = {}
    specimen_table = frame[["specimen_id", "scientific_name"]].drop_duplicates()
    for _, group in specimen_table.groupby("scientific_name", sort=True):
        specimens = group["specimen_id"].astype(str).to_numpy(copy=True)
        rng.shuffle(specimens)
        validation_count, test_count = _partition_counts(
            len(specimens), validation_fraction, test_fraction
        )
        for specimen_id in specimens[:test_count]:
            specimen_split[specimen_id] = "test"
        for specimen_id in specimens[test_count : test_count + validation_count]:
            specimen_split[specimen_id] = "validation"
        for specimen_id in specimens[test_count + validation_count :]:
            specimen_split[specimen_id] = "train"

    result = frame.copy()
    result["split"] = result["specimen_id"].astype(str).map(specimen_split)
    return result


def summarize_counts(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return frame.groupby(list(columns), dropna=False).size().rename("images").reset_index()

