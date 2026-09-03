# Standardized specimen classifier

## Scope

The classifier accepts only standardized, segmented specimen photographs. It does not identify butterflies in natural scenes. Scale bars, labels, colour targets, filenames and taxonomic text are excluded from model pixels and metadata inputs.

## Required manifest

One row per image:

| Column | Required | Meaning |
|---|---:|---|
| `image_path` | yes | Path to the image |
| `specimen_id` | yes | Stable individual specimen ID |
| `scientific_name` | yes | Expert-validated accepted name |
| `genus` | yes | Accepted genus |
| `view_side` | yes | `dorsal`, `ventral`, or `unknown` |
| `institution_code` | yes | Holding institution |
| `imaging_batch` | yes | Camera/lighting batch |
| `split` | yes | `train`, `validation`, or `test` |

## Leakage rules

1. All views of one specimen remain in one split.
2. Derived crops and augmented images inherit the parent split.
3. No label, ruler or colour-card pixels enter the model.
4. Filenames and directories do not contain taxon names in deployed inference inputs.
5. The final test set should include institutions or imaging batches not used for parameter selection.

## Baselines

Train and report all three under the same split:

1. BioCLIP 2/2.5 encoder with a linear species head, then partial fine-tuning.
2. DINO encoder with a linear head.
3. ConvNeXt-V2 supervised transfer-learning baseline.

Use a hierarchical genus/species loss, class-balanced sampling, calibrated probabilities and an explicit rejection threshold. Report macro-F1, top-1, top-5, per-species recall, genus accuracy, expected calibration error and unknown-species rejection performance.

## Minimum experimental sequence

1. Freeze encoders and benchmark linear probes.
2. Fine-tune the strongest encoder with identical augmentations and splits.
3. Compare full-specimen input with full specimen plus forewing/hindwing crops.
4. Evaluate dorsal and ventral subsets separately.
5. Run leave-one-institution or leave-one-batch-out evaluation.
6. Select the rejection threshold on validation data only.

