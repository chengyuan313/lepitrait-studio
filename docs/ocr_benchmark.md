# OCR benchmark

## v0.2 pilot set

The first real-image check used nine 1440 × 960 museum specimen frames supplied for local
testing. The source images are not included in this repository.

The images contain a useful mix of printed and handwritten labels, multiple labels per frame,
type-status discs, collection labels, catalogue labels, matrix codes and a millimetre ruler.
Labels are positioned to the right of the specimen, matching the current capture-layout
assumption.

| Field | Pilot result | Interpretation |
|---|---:|---|
| Catalogue number | 9/9 | Suitable for automatic prefill with human confirmation |
| Type wording | Detected when OCR rendered `Type` | Keep as a reviewable candidate |
| Complete event date | 0 automatically parsed | Historical handwriting is not reliable with the current engine |
| Locality and collector | Partially transcribed | Preserve verbatim OCR, then correct manually |
| Scientific name | Inconsistent | Do not accept without taxonomic review |

The catalogue-number parser corrects two observed OCR artefacts: embedded whitespace and `O`
in place of zero after the `NHMUK` prefix. This normalization is deliberately restricted to the
catalogue-number field.

## Acceptance rule

OCR output remains editable evidence. A record is not considered reviewed until a person has
checked the image, verbatim transcription and structured fields. OCR text, labels, rulers and
matrix codes are excluded from all species-classifier inputs.

## Next benchmark

Expand to at least 100 specimens stratified by museum, imaging batch, printed versus handwritten
content, label rotation and image resolution. Report character error rate for transcription and
field-level precision/recall separately; a single OCR accuracy number would conceal the current
gap between printed catalogue labels and handwriting.
