"""Validate specimen-level split integrity before any model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "image_path",
    "specimen_id",
    "scientific_name",
    "genus",
    "view_side",
    "institution_code",
    "imaging_batch",
    "split",
}
ALLOWED_SPLITS = {"train", "validation", "test"}
ALLOWED_VIEWS = {"dorsal", "ventral", "unknown"}


def validate_manifest(frame: pd.DataFrame, require_files: bool = True) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        return [f"Missing columns: {', '.join(sorted(missing))}"]

    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        errors.append("Required columns contain empty values.")
    unknown_splits = set(frame["split"].dropna()) - ALLOWED_SPLITS
    if unknown_splits:
        errors.append(f"Unknown split values: {sorted(unknown_splits)}")
    unknown_views = set(frame["view_side"].dropna()) - ALLOWED_VIEWS
    if unknown_views:
        errors.append(f"Unknown view values: {sorted(unknown_views)}")

    split_counts = frame.groupby("specimen_id")["split"].nunique()
    leaking = split_counts[split_counts > 1].index.tolist()
    if leaking:
        errors.append(f"Specimen leakage across splits: {leaking[:10]}")

    inconsistent = frame.groupby("specimen_id")["scientific_name"].nunique()
    inconsistent_ids = inconsistent[inconsistent > 1].index.tolist()
    if inconsistent_ids:
        errors.append(f"Specimens with multiple species labels: {inconsistent_ids[:10]}")

    duplicate_paths = frame[frame["image_path"].duplicated()]["image_path"].tolist()
    if duplicate_paths:
        errors.append(f"Duplicate image paths: {duplicate_paths[:10]}")

    if require_files:
        absent = [path for path in frame["image_path"] if not Path(path).is_file()]
        if absent:
            errors.append(f"Missing image files: {absent[:10]}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    errors = validate_manifest(frame, require_files=not args.skip_file_check)
    if errors:
        raise SystemExit("Manifest invalid:\n- " + "\n- ".join(errors))
    print(f"Manifest valid: {len(frame)} images, {frame['specimen_id'].nunique()} specimens")


if __name__ == "__main__":
    main()

