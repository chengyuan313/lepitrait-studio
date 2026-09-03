# Dataset contract

Every image is described by one CSV row. The manifest, rather than the folder name, is the
source of truth.

| Column | Meaning |
|---|---|
| `image_id` | Globally unique image identifier |
| `specimen_id` | Biological individual; all views share this value |
| `image_path` | Local classifier-ready RGB image |
| `scientific_name` | Expert-reviewed accepted binomial |
| `genus` | Accepted genus |
| `family` | Accepted family |
| `view` | `dorsal`, `ventral`, `lateral`, or `unknown` |
| `domain` | `museum_standardized`, `field_standardized`, or `field_in_situ` |
| `dataset_source` | Dataset or institution provenance |
| `license` | Image reuse licence |
| `label_pixels_removed` | Must be true; confirms no visible answer text |
| `split` | Added by `eurolepi split` |

Recommended optional columns are `taxon_id`, `country_code`, `photographer_id`,
`collection_batch`, `sex`, `life_stage`, `expert_verified`, and `original_record_id`.

## Inclusion rules

- Use adult butterflies with species-level expert labels.
- Preserve difficult examples, but flag severe occlusion and damage in optional columns.
- Keep dorsal and ventral photographs of one specimen under one `specimen_id`.
- Resolve synonyms before training while retaining the original name in an optional column.
- Exclude species complexes that cannot be separated visually, or train them as a named
  complex rather than inventing species-level certainty.

## Domain rule

A model intended for live field recognition needs field photographs in its training and test
sets. Museum accuracy is not evidence of field accuracy. Keep domain labels even when all
images initially come from one source, so later expansion remains measurable.

