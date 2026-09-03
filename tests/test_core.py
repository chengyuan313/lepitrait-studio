import unittest

import numpy as np
import pandas as pd

from PIL import Image

from lepitrait.metadata import parse_label, suggested_label_crop
from lepitrait.pipeline import AnalysisPipeline
from lepitrait.schema import ViewSide
from training.validate_manifest import validate_manifest


class CorePipelineTests(unittest.TestCase):
    def specimen_image(self):
        image = np.full((220, 300, 3), 248, dtype=np.uint8)
        yy, xx = np.ogrid[:220, :300]
        left = ((xx - 105) / 58) ** 2 + ((yy - 105) / 70) ** 2 < 1
        right = ((xx - 195) / 58) ** 2 + ((yy - 105) / 70) ** 2 < 1
        body = (np.abs(xx - 150) < 10) & (np.abs(yy - 115) < 76)
        image[left | right | body] = [72, 49, 31]
        return image

    def test_pipeline_produces_scale_and_colour_traits(self):
        result = AnalysisPipeline().analyse(
            self.specimen_image(),
            specimen_id="TEST-1",
            image_name="test_dorsal.tif",
            pixels_per_mm=10,
            colour_calibrated=True,
            view_side=ViewSide.DORSAL,
        )
        names_and_units = {(m.name, m.unit) for m in result.record.measurements}
        self.assertIn(("silhouette_area", "mm2"), names_and_units)
        self.assertTrue(result.record.colour[0].calibrated)
        self.assertGreater(result.segmentation.foreground_fraction, 0.03)

    def test_missing_scale_is_explicit(self):
        result = AnalysisPipeline().analyse(self.specimen_image(), "TEST-2", "test.tif")
        self.assertIn("SCALE_MISSING", {flag.code for flag in result.record.quality_flags})
        self.assertIn("COLOUR_UNCALIBRATED", {flag.code for flag in result.record.quality_flags})

    def test_label_parser_is_conservative(self):
        metadata = parse_label("Parnassius apollo 2019-07-21 60.17, 24.94")
        self.assertEqual(metadata.verbatim_scientific_name, "Parnassius apollo")
        self.assertEqual(metadata.event_date.isoformat(), "2019-07-21")
        self.assertAlmostEqual(metadata.decimal_latitude, 60.17)

    def test_museum_label_parser_reads_catalog_date_and_type(self):
        metadata = parse_label(
            "Baguio, subprov. Benguet\n31. iii. 1912\nHOLO-TYPE\nNHMUK016480156"
        )
        self.assertEqual(metadata.catalog_number, "NHMUK016480156")
        self.assertEqual(metadata.institution_code, "NHMUK")
        self.assertEqual(metadata.event_date.isoformat(), "1912-03-31")
        self.assertEqual(metadata.type_status, "holotype")

    def test_label_crop_matches_standard_right_panel_layout(self):
        crop = suggested_label_crop(Image.new("RGB", (1000, 500)))
        self.assertEqual(crop.box, (560, 0, 1000, 410))

    def test_manifest_rejects_specimen_leakage(self):
        frame = pd.DataFrame(
            [
                {"image_path": "a.tif", "specimen_id": "A", "scientific_name": "Pieris napi", "genus": "Pieris", "view_side": "dorsal", "institution_code": "MZH", "imaging_batch": "1", "split": "train"},
                {"image_path": "b.tif", "specimen_id": "A", "scientific_name": "Pieris napi", "genus": "Pieris", "view_side": "ventral", "institution_code": "MZH", "imaging_batch": "1", "split": "test"},
            ]
        )
        errors = validate_manifest(frame, require_files=False)
        self.assertTrue(any("leakage" in error.lower() for error in errors))


if __name__ == "__main__":
    unittest.main()
