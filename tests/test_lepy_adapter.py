from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
import zipfile

from PIL import Image

from eurolepi.lepy_adapter import (
    LepyAdapter,
    LepySettings,
    inspect_field_metadata,
    pair_trait_uploads,
)


def image_bytes(colour: str = "white", image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), colour).save(buffer, format=image_format)
    return buffer.getvalue()


class LepyAdapterTests(unittest.TestCase):
    def _fake_lepy(self, root: Path) -> LepySettings:
        home = root / "LEPY"
        home.mkdir()
        (home / "config.yml").write_text(
            textwrap.dedent(
                """
                reading:
                  rgb_regex: '(.+)\\.tif'
                  uv_regex: '(.+)uv\\.tif'
                  uv_channel_index: 0
                  extensions: [.jpg]
                  ordered: false
                segmentation:
                  method: flatbug
                calibration:
                  enabled: false
                points_of_interest:
                  enabled: true
                execution:
                  proceed: true
                  force: true
                """
            ).strip(),
            encoding="utf-8",
        )
        (home / "main.py").write_text(
            textwrap.dedent(
                """
                import argparse
                import csv
                from pathlib import Path

                parser = argparse.ArgumentParser()
                parser.add_argument('folder')
                parser.add_argument('config')
                parser.add_argument('--output')
                parser.add_argument('--yes', action='store_true')
                parser.add_argument('--force', action='store_true')
                parser.add_argument('--n_jobs')
                args = parser.parse_args()
                output = Path(args.output)
                output.mkdir(parents=True, exist_ok=True)
                (output / 'visualisations').mkdir()
                fields = [
                    'Code', 'contour_width_calibrated', 'contour_height_calibrated',
                    'contour_area_calibrated', 'poi_dist_inner_outer_l',
                    'poi_dist_inner_outer_r', 'poi_dist_inner', 'poi_dist_body',
                    'poi_area_body', 'poi_area_wing_l', 'poi_area_wing_r',
                    'luminance_mean', 'red_median', 'green_median', 'blue_median'
                ]
                with (output / 'stats.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\\t')
                    writer.writeheader()
                    for image in sorted(Path(args.folder).glob('*_rgb.*')):
                        code = image.stem
                        writer.writerow({
                            'Code': code, 'contour_width_calibrated': 44.5,
                            'contour_height_calibrated': 20.1,
                            'contour_area_calibrated': 900.2,
                            'poi_dist_inner_outer_l': 21.2,
                            'poi_dist_inner_outer_r': 21.0,
                            'poi_dist_inner': 3.5, 'poi_dist_body': 18.4,
                            'poi_area_body': 55.0, 'poi_area_wing_l': 410.0,
                            'poi_area_wing_r': 408.0, 'luminance_mean': 122.0,
                            'red_median': 100, 'green_median': 110, 'blue_median': 120,
                        })
                        (output / 'visualisations' / f'{code}.png').write_bytes(b'preview')
                (output / 'errors.log').write_text('')
                """
            ).strip(),
            encoding="utf-8",
        )
        return LepySettings(home, sys.executable, home / "config.yml", timeout_seconds=20)

    def test_pairs_rgb_and_uv_files(self):
        samples = pair_trait_uploads(
            [
                ("site/A01_rgb.png", image_bytes("white")),
                ("site/A01_uv.png", image_bytes("black")),
                ("site/A02.png", image_bytes("blue")),
            ]
        )
        self.assertEqual([sample.specimen_id for sample in samples], ["A01", "A02"])
        self.assertIsNotNone(samples[0].uv_bytes)
        self.assertIsNone(samples[1].uv_bytes)

    def test_metadata_validation_and_mapping(self):
        payload = (
            b"specimen_id,site_id,latitude,longitude,collection_date,temperature_c\n"
            b"A01,FI-01,60.1699,24.9384,2027-06-14,18.2\n"
        )
        inspection = inspect_field_metadata(payload, ["A01", "A02"])
        self.assertTrue(inspection.valid)
        self.assertEqual(len(inspection.warnings), 1)
        self.assertEqual(inspection.table.loc[0, "site_id"], "FI-01")

    def test_rejects_duplicate_rgb_images(self):
        with self.assertRaisesRegex(ValueError, "Duplicate RGB"):
            pair_trait_uploads(
                [("A01.jpg", image_bytes()), ("A01_rgb.png", image_bytes())]
            )

    def test_runs_external_lepy_and_normalises_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = LepyAdapter(self._fake_lepy(root))
            samples = pair_trait_uploads([("A01_rgb.png", image_bytes())])
            metadata_csv = b"specimen_id,site_id,temperature_c\nA01,FI-01,18.2\n"
            metadata = inspect_field_metadata(metadata_csv, ["A01"]).table
            result = adapter.run(samples, metadata)

            self.assertEqual(result.table.loc[0, "status"], "completed")
            self.assertEqual(result.table.loc[0, "body_length_mm"], 18.4)
            self.assertEqual(result.table.loc[0, "site_id"], "FI-01")
            self.assertIn("A01", result.visualisations)
            with zipfile.ZipFile(BytesIO(result.archive_bytes)) as archive:
                self.assertIn("trait_results.csv", archive.namelist())
                self.assertIn("run_manifest.json", archive.namelist())


if __name__ == "__main__":
    unittest.main()
