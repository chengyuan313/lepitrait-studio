# Training protocol

## Baseline

The default configuration fine-tunes `maxvit_tiny_tf_224.in1k`, the `timm` MaxViT-T variant
pretrained on ImageNet-1K. It uses 224 px images, AdamW, cosine learning-rate decay, label
smoothing, balanced sampling, mixed precision on CUDA, and early stopping on validation
macro-F1.

These settings are a reproducible baseline, not a claimed optimum. Record every config and
software version used for a reported result.

## Sequence

1. Freeze the species list and accepted taxonomy version.
2. Validate the unsplit manifest.
3. Split by `specimen_id` within species.
4. Train the MaxViT-T baseline.
5. Select a rejection threshold on validation data.
6. Evaluate once on the held-out test set.
7. Report results by species and image domain.
8. Save the checkpoint, class order, history and model card together.

## Class imbalance

Balanced sampling prevents abundant species from dominating gradient updates. It does not
create missing morphological variation. Species with very few independent specimens should be
returned at genus level or routed to expert review rather than advertised as automated species
identifications.

## Field expansion

When field images become available, append them to the manifest with `domain=field_in_situ`,
then create a new versioned split and retrain. Do not overwrite the original museum-only test
set; keep it as a regression benchmark.

