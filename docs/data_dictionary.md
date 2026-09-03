# Data dictionary

The canonical record is `SpecimenRecord` in `src/lepitrait/schema.py`.

| Group | Key fields | Purpose |
|---|---|---|
| Identity | `specimen_id`, `schema_version` | Stable join and schema migration |
| Capture | image, view, equipment, batch, scale, profile | Imaging provenance and batch effects |
| Metadata | Darwin Core-oriented label fields | Time, place, collector and accepted taxonomy |
| Measurements | name, value, unit, method, region | Long-form morphological traits |
| Colour | CIELAB, chroma, hue, dark fraction, calibration flag | Comparable pigmentation summaries |
| Predictions | taxon, rank, probability, model/version | Top-k identification evidence |
| QC | code, severity, message, observed value | Machine-readable inclusion filters |
| Provenance | pipeline/model versions and parameters | Reproducibility |

The flat CSV export prefixes fields by group and is intended for R. JSON remains the authoritative nested record.

