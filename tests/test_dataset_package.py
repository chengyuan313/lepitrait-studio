from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from eurolepi.dataset_package import (
    inspect_dataset_zip,
    install_dataset,
    specimen_safe_split,
)


SPECIES = (
    "Pieris napi",
    "Vanessa atalanta",
    "Aglais io",
    "Gonepteryx rhamni",
    "Polyommatus icarus",
)


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (18, 14), (120, 80, 45)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_dataset_zip(
    species: tuple[str, ...] = SPECIES,
    specimens_per_species: int = 3,
    extra_file: tuple[str, bytes] | None = None,
) -> bytes:
    rows = []
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for species_index, scientific_name in enumerate(species):
            for specimen_index in range(specimens_per_species):
                specimen_id = f"S{species_index}-{specimen_index}"
                path = f"images/{specimen_id}.jpg"
                rows.append(
                    {
                        "image_path": path,
                        "scientific_name": scientific_name,
                        "specimen_id": specimen_id,
                    }
                )
                archive.writestr(path, image_bytes())
        archive.writestr("manifest.csv", pd.DataFrame(rows).to_csv(index=False))
        if extra_file:
            archive.writestr(*extra_file)
    return buffer.getvalue()


class DatasetPackageTests(unittest.TestCase):
    def test_valid_package_passes(self):
        report = inspect_dataset_zip(make_dataset_zip())
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.image_count, 15)
        self.assertEqual(report.specimen_count, 15)
        self.assertEqual(report.species_count, 5)

    def test_requires_five_species(self):
        report = inspect_dataset_zip(make_dataset_zip(species=SPECIES[:4]))
        self.assertFalse(report.valid)
        self.assertTrue(any("At least 5 species" in error for error in report.errors))

    def test_rejects_unrelated_files(self):
        report = inspect_dataset_zip(make_dataset_zip(extra_file=("notes.xlsx", b"not a sheet")))
        self.assertFalse(report.valid)
        self.assertTrue(any("Unsupported files" in error for error in report.errors))

    def test_specimen_split_never_leaks(self):
        report = inspect_dataset_zip(make_dataset_zip())
        split = specimen_safe_split(report.manifest)
        split_counts = split.groupby("specimen_id")["split"].nunique()
        self.assertEqual(int(split_counts.max()), 1)
        self.assertEqual(set(split["split"]), {"train", "validation"})

    def test_install_extracts_only_referenced_images(self):
        blob = make_dataset_zip()
        with tempfile.TemporaryDirectory() as directory:
            installed = install_dataset(blob, "European pilot", Path(directory))
            self.assertTrue(installed.manifest_path.is_file())
            self.assertEqual(len(list((installed.path / "images").glob("*.jpg"))), 15)


if __name__ == "__main__":
    unittest.main()

