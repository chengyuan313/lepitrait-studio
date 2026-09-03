# Evaluation protocol

Always report Top-1, Top-5 and macro-F1. Also produce per-species precision/recall and a
confusion matrix during a real study analysis.

## Required subsets

- museum standardized images;
- standardised field specimens, if available;
- natural in-situ field images, if available;
- dorsal and ventral views;
- common and rare species;
- intact and damaged/occluded specimens.

## Rejection test

Softmax confidence alone does not prove that an image belongs to a trained species. Build an
open-set test containing European species deliberately excluded from training and non-target
insects. Select the rejection method and threshold using validation data, then report unknown
detection AUROC/OSCR in addition to closed-set accuracy.

The current GUI implements a transparent confidence threshold as an MVP. Before scientific or
operational deployment, compare it with energy scores or distance-to-class-prototype methods.

## Leakage audit

- no specimen occurs in multiple splits;
- no near-duplicate burst images cross splits;
- no label, QR code, catalogue number, filename text or folder name appears in model pixels;
- photographer and collection batch performance are inspected separately;
- the final test set is not used to tune the threshold.

