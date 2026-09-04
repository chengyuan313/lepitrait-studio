# EuroLepi Studio

EuroLepi Studio is an English-language research application for standardized European butterfly
identification and field-specimen trait extraction. It supports the image-analysis part of the
THERMOTRAIT workflow: identify specimens, quantify morphology and pigmentation, and retain the
field metadata needed to compare regions and climates.

The GUI contains five independent workflows:

1. **Train Model** — validate a labelled image package and train a MaxViT classifier.
2. **Single Identification** — return Top-5 species with probabilities and reference images.
3. **Batch Identification** — identify an uploaded image folder and export a Top-5 CSV.
4. **Single Trait Extraction** — run LEPY on one standardized RGB/optional UV specimen pair.
5. **Batch Trait Extraction** — pair a folder of specimens, run LEPY, join field metadata, and
   export the complete trait table and analysis artifacts.

The software does not infer climate from an image. `site_id`, coordinates, collection date, and
field temperature are explicit metadata joined by `specimen_id`, so image measurements remain
traceable to the biological sample and sampling event.

## Install and open the GUI

Python 3.12 is recommended. In PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

## Identification training data

The **Train Model** page accepts one `.zip` file with this exact structure:

```text
my-european-butterflies.zip
├── manifest.csv
└── images/
    ├── specimen_0001_dorsal.jpg
    ├── specimen_0001_ventral.jpg
    ├── specimen_0002.jpg
    └── ...
```

Only `.jpg`, `.jpeg`, and `.png` images are accepted for classification. `manifest.csv` must be
UTF-8 with one row per image:

```csv
image_path,scientific_name,specimen_id
images/specimen_0001_dorsal.jpg,Pieris napi,NHM-0001
images/specimen_0001_ventral.jpg,Pieris napi,NHM-0001
images/specimen_0002.jpg,Vanessa atalanta,NHM-0002
```

| Column | Required | Meaning |
|---|---:|---|
| `image_path` | Yes | Relative image path inside `images/` |
| `scientific_name` | Yes | Expert-verified species used as the class label |
| `specimen_id` | Yes | Biological individual ID; all views share this ID |
| `common_name` | No | Display-only common name |

Blocking checks enforce at least five species and three distinct specimens per species, readable
images, unique paths, and a one-species-per-specimen rule. All views of one specimen stay in the
same train/validation split. Crop out labels, QR codes, catalogue numbers, rulers, and colour cards
so the classifier cannot learn collection artifacts. Twenty or more specimens per species is a
reasonable pilot target; field deployment requires representative field images and an independent
held-out test set.

Training writes `model.json`, `best.pt`, `history.csv`, and one display reference per species under
`workspace/models/<model-id>/`. The classifier uses `maxvit_tiny_tf_224.in1k` from `timm`, ImageNet
pretraining, class-balanced sampling, specimen-safe splitting, AdamW, and label smoothing.

## LEPY installation

Trait pages call the official [tzlr-de/LEPY](https://github.com/tzlr-de/LEPY) command-line program.
LEPY is not copied into this repository: it has its own GPL-3.0 licence, dependency versions, model
files, and release cycle. EuroLepi Studio runs it as a separate process and records a fingerprint of
the selected engine plus a SHA-256 hash of the effective configuration.

On Windows, download LEPY with **Code → Download ZIP**, extract it to a separate folder, and follow
its official installation instructions in its own virtual environment. LEPY documents Python 3.11+
and pinned dependencies. Then enter these three paths on either trait page:

```text
LEPY installation folder    C:\research-tools\LEPY
LEPY Python executable      C:\research-tools\LEPY\.venv\Scripts\python.exe
LEPY configuration          C:\research-tools\LEPY\config.yml
```

The adapter validates `main.py`, the YAML configuration, the interpreter, and the scale-bar
template before enabling a run. It invokes:

```text
<lepy-python> <lepy-home>/main.py <staged-input> <job-config>
  --output <job-output> --yes --force --n_jobs N
```

It never imports LEPY internals and never invokes a shell.

## Standardized trait images

LEPY is designed for a spread specimen on a controlled, light background with a detectable scale
bar. For comparable field data, keep the camera, lens, distance, exposure, light geometry,
background, scale-bar design, specimen orientation, and dorsal/ventral protocol constant across
sites. A colour workflow is only comparable across batches when capture and calibration are held
constant. RGB is required; registered UV is optional.

Use [`docs/FIELD_IMAGING_SOP.md`](docs/FIELD_IMAGING_SOP.md) as the field and imaging checklist.

### Single image

Open **Single Trait Extraction**, configure the LEPY engine, provide a stable specimen ID, and
upload one RGB image plus an optional UV image. Optional sampling metadata can be entered in the
same page. The result includes the LEPY visualization, a normalized CSV, and a ZIP containing all
official LEPY outputs.

### Batch folder

Use one shared basename for each biological individual:

```text
trait-images/
├── FIN-HEL-0001_rgb.tif
├── FIN-HEL-0001_uv.tif
├── FIN-HEL-0002_rgb.tif
└── ESP-SIE-0001_rgb.tif
```

Files without `_rgb` are also treated as RGB, but explicit suffixes are recommended. Supported
formats are JPG, JPEG, PNG, TIF, and TIFF. Every `_uv` image must have one RGB partner. Duplicate
specimen/modality pairs are rejected before LEPY runs.

An optional UTF-8 metadata CSV maps sampling and identification data by `specimen_id`:

```csv
specimen_id,scientific_name,site_id,country,latitude,longitude,collection_date,temperature_c
FIN-HEL-0001,Pieris napi,FI-HEL-01,Finland,60.1699,24.9384,2027-06-14,18.2
FIN-HEL-0002,Vanessa atalanta,FI-HEL-01,Finland,60.1699,24.9384,2027-06-14,18.2
ESP-SIE-0001,Pieris napi,ES-SIE-03,Spain,37.0950,-3.3930,2027-07-02,24.7
```

Only these columns are accepted:

| Column | Required | Meaning |
|---|---:|---|
| `specimen_id` | Yes when a metadata CSV is supplied | Exact basename ID used by the image pair |
| `scientific_name` | No | Expert/model-reviewed identification |
| `site_id` | No | Stable sampling-site code |
| `country` | No | Country name or project code |
| `latitude`, `longitude` | No | Decimal WGS84 coordinates |
| `collection_date` | No | ISO date, for example `2027-06-14` |
| `temperature_c` | No | Observed field temperature in °C |

Unknown columns, duplicate IDs, invalid coordinates, invalid dates, and non-numeric temperatures
are reported before processing. Missing and extra IDs are shown as warnings.

## Trait outputs

The normalized CSV retains every LEPY column and adds provenance, metadata, and stable aliases:

| Stable output | LEPY source |
|---|---|
| `wing_span_mm` | `contour_width_calibrated` |
| `body_bbox_length_mm` | `contour_height_calibrated` |
| `specimen_area_mm2` | `contour_area_calibrated` |
| `left_forewing_length_mm` | `poi_dist_inner_outer_l` |
| `right_forewing_length_mm` | `poi_dist_inner_outer_r` |
| `body_width_mm` | `poi_dist_inner` |
| `body_length_mm` | `poi_dist_body` |
| `body_area_mm2` | `poi_area_body` |
| `left_wing_area_mm2`, `right_wing_area_mm2` | LEPY POI wing areas |

Raw colour columns include RGB and UV summaries, HSV intensity/saturation/hue, luminance,
chromaticity, IQR contrast, and Shannon/Simpson colour diversity where LEPY produces them. Each row
also includes `status`, `error`, processing time, engine fingerprint, configuration hash, original
filename, and optional field metadata. The ZIP retains `stats.csv`, JSON, contours,
visualisations, errors, the effective config, normalized `trait_results.csv`, and `run_manifest.json`.

## Command line

```powershell
# Identification
.\.venv\Scripts\python.exe -m eurolepi.cli validate path\to\dataset.zip
.\.venv\Scripts\python.exe -m eurolepi.cli train path\to\dataset.zip --dataset-name "European butterflies 2027"
.\.venv\Scripts\python.exe -m eurolepi.cli predict MODEL_ID path\to\image.jpg
.\.venv\Scripts\python.exe -m eurolepi.cli batch MODEL_ID path\to\folder --output identification_results.csv

# Traits
.\.venv\Scripts\python.exe -m eurolepi.cli traits-single specimen.tif `
  --specimen-id FIN-HEL-0001 --lepy-home C:\research-tools\LEPY `
  --lepy-python C:\research-tools\LEPY\.venv\Scripts\python.exe

.\.venv\Scripts\python.exe -m eurolepi.cli traits-batch path\to\trait-images `
  --metadata field_metadata.csv --lepy-home C:\research-tools\LEPY `
  --lepy-python C:\research-tools\LEPY\.venv\Scripts\python.exe
```

Use the global `--workspace` option before the command to select another model workspace.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The adapter tests use a deterministic fake command-line engine to verify staging, subprocess
isolation, column normalization, metadata joins, and artifact packaging. They do not claim to
validate LEPY's biological accuracy; that requires the official LEPY installation and a manually
measured, held-out specimen set from the target imaging protocol.
