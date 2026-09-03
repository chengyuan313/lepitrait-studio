from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from eurolepi.batch import RESULT_COLUMNS, identify_batch
from eurolepi.registry import ModelRegistry
from eurolepi.types import Candidate, IdentificationResult


class FakeIdentifier:
    def predict(self, image: Image.Image, top_k: int = 5):
        candidates = tuple(
            Candidate(rank, f"Species example{rank}", 1.0 / (rank + 1))
            for rank in range(1, top_k + 1)
        )
        return IdentificationResult(candidates)


class RegistryAndBatchTests(unittest.TestCase):
    def test_registry_lists_dataset_name(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "pilot-model"
            model_path.mkdir()
            (model_path / "best.pt").write_bytes(b"checkpoint")
            (model_path / "model.json").write_text(
                json.dumps(
                    {
                        "model_id": "pilot-model",
                        "dataset_name": "European pilot",
                        "created_at": "2027-01-02T03:04:05+00:00",
                        "backbone": "maxvit_tiny_tf_224.in1k",
                        "species": ["Pieris napi"] * 5,
                        "references": {},
                    }
                ),
                encoding="utf-8",
            )
            records = ModelRegistry(Path(directory)).list_models()
            self.assertEqual(len(records), 1)
            self.assertIn("European pilot", records[0].display_name)

    def test_batch_returns_fixed_top_five_columns(self):
        from io import BytesIO

        buffer = BytesIO()
        Image.new("RGB", (12, 12), "white").save(buffer, format="PNG")
        result = identify_batch([("folder/butterfly.png", buffer.getvalue())], FakeIdentifier())
        self.assertEqual(list(result.columns), RESULT_COLUMNS)
        self.assertEqual(result.loc[0, "image_name"], "folder/butterfly.png")
        self.assertEqual(result.loc[0, "top5_species"], "Species example5")
        self.assertEqual(result.loc[0, "error"], "")

    def test_batch_keeps_unreadable_file_as_error_row(self):
        result = identify_batch([("broken.jpg", b"not an image")], FakeIdentifier())
        self.assertEqual(len(result), 1)
        self.assertNotEqual(result.loc[0, "error"], "")


if __name__ == "__main__":
    unittest.main()

