# EuroLepi ID

EuroLepi ID is a training and inference framework for image-based identification of European
butterflies. The repository has one scope: prepare a labelled image dataset, fine-tune a
MaxViT-T classifier, evaluate each image domain separately, and return reviewable Top-k
predictions with a low-confidence rejection decision.

This repository replaces the former LEPY/OCR/trait-extraction prototype. It contains no label
OCR, morphometrics, colour analysis, climate matching or LEPY adapter.

## Design boundary

- Primary task: European butterfly species classification.
- Inputs: label-free museum images, standardised field specimens, and/or in-situ field images.
- Default model: `maxvit_tiny_tf_224.in1k` from `timm`, following the MaxViT-T approach tested
  by Barkmann, Lindner & Rüdisser (2026).
- Output: Top-5 candidates plus `unknown_or_review` below a configurable threshold.
- Split unit: specimen, never image.
- Museum labels, QR codes, catalogue numbers and rulers are forbidden classifier pixels.
- Accuracy is reported separately for museum and field domains.

## Install

The dataset checker and GUI require Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
streamlit run app.py
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Install the ML stack only on the training or inference machine:

```bash
python -m pip install -e ".[ml]"
```

## Train when the European dataset is available

1. Copy `data/manifest.example.csv` to `data/manifest_unsplit.csv` and add one row per image.
2. Make every `image_path` point to an image that is safe for the classifier. For museum
   images, remove all label, QR-code, catalogue-number and ruler pixels.
3. Validate and create deterministic specimen-safe splits:

```bash
eurolepi validate data/manifest_unsplit.csv --before-split
eurolepi split data/manifest_unsplit.csv --output data/manifest.csv
eurolepi validate data/manifest.csv
```

4. Train and evaluate:

```bash
eurolepi train configs/maxvit_tiny.yaml
eurolepi evaluate models/eurolepi_maxvit_tiny/best.pt data/manifest.csv
```

5. Run one prediction:

```bash
eurolepi predict models/eurolepi_maxvit_tiny/best.pt butterfly.jpg
```

The trainer writes `best.pt`, `class_names.json`, `history.json`, and `model_card.json`.
Model weights and raw datasets are ignored by Git.

## Repository map

```text
app.py                         Streamlit identification and dataset-check GUI
configs/maxvit_tiny.yaml       Reproducible training configuration
data/manifest.example.csv      Dataset contract example
src/eurolepi/manifest.py       Validation and specimen-safe splitting
src/eurolepi/training.py       Balanced MaxViT-T fine-tuning
src/eurolepi/inference.py      Top-k prediction and rejection
src/eurolepi/metrics.py        Accuracy and macro-F1 metrics
docs/                          Dataset, training and evaluation protocols
tests/                         Dependency-light integrity tests
```

Read [the dataset contract](docs/dataset_contract.md),
[training protocol](docs/training_protocol.md), and
[evaluation protocol](docs/evaluation_protocol.md) before using biological data.

## Reference implementation

The default architecture follows the public MaxViT-T butterfly work:

- Barkmann F, Lindner A, Rüdisser J. 2026. *Comparing deep learning models for butterfly and
  moth species identification*. DOI: `10.1111/icad.70123`.
- Public Austrian model: <https://huggingface.co/RikeB/MaxViT_butterfly_identification>

The Austrian checkpoint is not bundled because it targets citizen-science photographs from a
limited Austrian species list. EuroLepi ID trains a new class head and checkpoint for the
user-supplied European dataset.
