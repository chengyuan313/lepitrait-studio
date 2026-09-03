# Architecture

LepiTrait Studio is a modular monolith. The Streamlit GUI, command-line entry point and future batch worker all call the same domain pipeline.

```text
standardized image
  -> capture QC
  -> LEPY adapter or transparent baseline
       -> specimen mask
       -> scale-aware morphology
       -> calibrated colour traits
  -> label OCR/parser (separate crop)
  -> specimen-only classifier
  -> reviewable SpecimenRecord
  -> JSON / flat CSV / Parquet in a later release
```

## Design rules

1. Raw files are immutable.
2. Every output carries method, version and parameters.
3. Automated results remain editable and the reviewed state is explicit.
4. Label pixels never reach the species classifier.
5. Uncalibrated colours and missing scale are represented as QC states, not silently accepted.
6. LEPY and species models are adapters, so upstream model changes do not rewrite the GUI or schema.

## Production extensions

- pin a validated LEPY revision and implement a concrete runner;
- add four-wing masks and landmark overlays from LEPY outputs;
- train BioCLIP 2/2.5, DINO and ConvNeXt baselines from the same manifest;
- implement institution/specimen-aware splits and open-set rejection;
- add GBIF/Catalogue of Life validation behind a cached taxonomy service;
- store large jobs in Parquet with model artifacts tracked outside Git.

