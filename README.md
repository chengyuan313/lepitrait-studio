# EuroLepi ID

EuroLepi ID is an English-language application for training and running European butterfly
image classifiers. It has exactly three workflows:

1. **Train Model** — upload a validated training dataset and create a model.
2. **Single Identification** — choose a trained dataset/model and identify one image.
3. **Batch Identification** — choose a model, upload an image folder, and download Top-5 CSV results.

No OCR, trait measurement, colour analysis, climate analysis, or geographic matching is included.

## Install and open the GUI

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

## Training dataset format

The **Train Model** page accepts one `.zip` file only. The ZIP must have this exact structure:

```text
my-european-butterflies.zip
├── manifest.csv
└── images/
    ├── specimen_0001_dorsal.jpg
    ├── specimen_0001_ventral.jpg
    ├── specimen_0002.jpg
    └── ...
```

Only `.jpg`, `.jpeg`, and `.png` images are accepted. The ZIP must not contain model files,
spreadsheets other than `manifest.csv`, hidden metadata, labels, QR-code crops, or unrelated files.

`manifest.csv` must be UTF-8 and contain one row per image:

```csv
image_path,scientific_name,specimen_id
images/specimen_0001_dorsal.jpg,Pieris napi,NHM-0001
images/specimen_0001_ventral.jpg,Pieris napi,NHM-0001
images/specimen_0002.jpg,Vanessa atalanta,NHM-0002
```

| Column | Required | Meaning |
|---|---:|---|
| `image_path` | Yes | Relative path inside `images/` in the ZIP |
| `scientific_name` | Yes | Expert-verified scientific name used as the class label |
| `specimen_id` | Yes | Biological individual ID; multiple views of one individual share this ID |
| `common_name` | No | Display name only; never used as the training target |

Validation rules:

- At least **5 species**.
- At least **3 distinct specimens per species**.
- Every manifest image must exist and be a readable JPG, JPEG, or PNG.
- One `specimen_id` cannot belong to more than one species.
- One image path cannot occur twice.
- All views of the same specimen stay in the same train/validation split.
- Museum images must be cropped so labels, QR codes, catalogue numbers, rulers, and colour cards
  are not visible to the classifier.

Twenty or more specimens per species is recommended for an initial experiment. Reliable field
identification requires field photographs in both training and held-out evaluation data.

An example manifest is available at [`examples/manifest.csv`](examples/manifest.csv).

## What training creates

Each run is stored under `workspace/models/<model-id>/`:

```text
model.json              dataset name, classes, backbone, and training metadata
best.pt                 best validation checkpoint
history.csv             epoch-level loss and accuracy
references/             one reference image per species for Top-5 display
```

The model selector in Single and Batch Identification lists these runs by their training dataset
name. Model files and uploaded datasets are intentionally excluded from Git.

## Single identification

1. Open **Single Identification**.
2. Select the training dataset/model to use.
3. Upload one JPG, JPEG, or PNG.
4. The app returns five ranked scientific names, confidence scores, and one stored reference image
   for each candidate species.

## Batch identification

1. Open **Batch Identification**.
2. Select the training dataset/model.
3. Upload a folder containing JPG, JPEG, or PNG images.
4. Run identification and download `identification_results.csv`.

The CSV contains `image_name` plus `top1_species` through `top5_species` and their probabilities.
An unreadable file is retained in the CSV with an `error` message rather than silently discarded.

## Command line

```powershell
.\.venv\Scripts\python.exe -m eurolepi.cli validate path\to\dataset.zip
.\.venv\Scripts\python.exe -m eurolepi.cli train path\to\dataset.zip --dataset-name "European butterflies 2027"
.\.venv\Scripts\python.exe -m eurolepi.cli predict MODEL_ID path\to\image.jpg
.\.venv\Scripts\python.exe -m eurolepi.cli batch MODEL_ID path\to\folder --output identification_results.csv
```

Use `--workspace` to select a workspace other than the default `workspace/` directory.

## Model

The default backbone is `maxvit_tiny_tf_224.in1k` from `timm`. Training uses ImageNet
pretraining, class-balanced sampling, specimen-safe train/validation splitting, AdamW, label
smoothing, and best-checkpoint selection by validation accuracy.
