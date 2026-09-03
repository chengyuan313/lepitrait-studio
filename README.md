# LepiTrait Studio

LepiTrait Studio is a local, GUI-first research workflow for standardized butterfly and moth specimen photographs. It is designed around the trait outputs described by **LEPY** and keeps every automated result reviewable and versioned.

The first release provides:

- image quality checks for standardized specimen photographs;
- a transparent white-background segmentation baseline;
- scale-aware morphology measurements;
- calibrated-image CIELAB colour summaries;
- specimen-label OCR hooks and conservative metadata parsing;
- a strict adapter boundary for LEPY and BioCLIP-based species identification;
- JSON/CSV export with provenance and quality-control flags;
- a Streamlit review interface.

The built-in segmentation is a development baseline, not a replacement for LEPY. Production analyses should connect a validated LEPY runner and compare automated measurements with manual ground truth.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
```

Run the command-line pipeline on one standardized image:

```bash
lepitrait specimen.jpg --specimen-id MZH-0001 --pixels-per-mm 42.8 --output result.json
```

Run tests without installing extra test packages:

```bash
python -m unittest discover -s tests -v
```

## Scientific boundary

Species identification is intentionally limited to standardized specimen photographs. Labels, rulers and colour charts must be excluded from the classifier input to prevent shortcut learning. Natural-scene photographs are outside the scope of this project.

Read [the imaging SOP](docs/imaging_sop.md), [architecture](docs/architecture.md), and [data dictionary](docs/data_dictionary.md) before collecting training data.

## LEPY integration

`lepitrait.lepy_adapter.LepyAdapter` is the stable boundary between this application and a locally installed LEPY workflow. The adapter expects a runner callable that receives an image path and output directory and returns a dictionary of measurements. This avoids copying or modifying upstream LEPY code and lets the project pin and validate a known upstream revision.

## Species model integration

`lepitrait.identification.SpeciesIdentifier` defines the inference contract. A production implementation should use a BioCLIP 2/2.5 visual encoder fine-tuned on label-free, standardized museum specimen crops. The GUI remains usable before model weights are installed and clearly marks identification as unavailable.

