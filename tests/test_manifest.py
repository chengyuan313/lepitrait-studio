import unittest

import pandas as pd

from eurolepi.manifest import assign_grouped_splits, validate_manifest


def rows(species_count=2, specimens_per_species=10):
    output = []
    names = ["Pieris napi", "Vanessa atalanta"][:species_count]
    for species in names:
        genus = species.split()[0]
        for number in range(specimens_per_species):
            specimen_id = f"{genus}-{number}"
            output.append(
                {
                    "image_id": f"{specimen_id}-d",
                    "specimen_id": specimen_id,
                    "image_path": f"{specimen_id}.jpg",
                    "scientific_name": species,
                    "genus": genus,
                    "family": "Nymphalidae" if genus == "Vanessa" else "Pieridae",
                    "view": "dorsal",
                    "domain": "museum_standardized",
                    "dataset_source": "test",
                    "license": "CC0",
                    "label_pixels_removed": True,
                }
            )
    return output


class ManifestTests(unittest.TestCase):
    def test_assign_splits_keeps_specimen_together(self):
        frame = pd.DataFrame(rows())
        second_views = frame.copy()
        second_views["image_id"] += "-v"
        second_views["image_path"] += "-v"
        second_views["view"] = "ventral"
        result = assign_grouped_splits(pd.concat([frame, second_views], ignore_index=True))
        self.assertEqual(result.groupby("specimen_id")["split"].nunique().max(), 1)
        self.assertEqual(set(result["split"]), {"train", "validation", "test"})

    def test_label_pixels_are_rejected(self):
        frame = pd.DataFrame(rows(species_count=1, specimens_per_species=4))
        frame.loc[0, "label_pixels_removed"] = False
        report = validate_manifest(frame, require_files=False, require_split=False)
        self.assertFalse(report.valid)
        self.assertTrue(any("label pixels" in item for item in report.errors))

    def test_split_leakage_is_rejected(self):
        frame = assign_grouped_splits(pd.DataFrame(rows()))
        duplicate = frame.iloc[[0]].copy()
        duplicate["image_id"] = "another-view"
        duplicate["image_path"] = "another-view.jpg"
        duplicate["split"] = "test" if duplicate.iloc[0]["split"] != "test" else "train"
        report = validate_manifest(
            pd.concat([frame, duplicate], ignore_index=True),
            require_files=False,
            minimum_train_images=1,
        )
        self.assertTrue(any("leakage" in item.lower() for item in report.errors))

    def test_invalid_domain_is_rejected(self):
        frame = pd.DataFrame(rows(species_count=1, specimens_per_species=4))
        frame.loc[0, "domain"] = "internet"
        report = validate_manifest(frame, require_files=False, require_split=False)
        self.assertFalse(report.valid)


if __name__ == "__main__":
    unittest.main()

