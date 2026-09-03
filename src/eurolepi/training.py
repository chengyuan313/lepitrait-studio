"""End-to-end MaxViT-T training from an installed dataset package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Callable
from uuid import uuid4

import pandas as pd
from PIL import Image

from eurolepi.dataset_package import InstalledDataset, install_dataset
from eurolepi.modeling import DEFAULT_BACKBONE, IMAGE_SIZE, create_model, image_transform, require_ml
from eurolepi.registry import write_model_metadata
from eurolepi.types import ModelRecord


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    backbone: str = DEFAULT_BACKBONE
    image_size: int = IMAGE_SIZE
    seed: int = 42


ProgressCallback = Callable[[int, int, dict[str, float]], None]


class _ButterflyDataset:
    def __init__(self, frame, dataset_path, classes, transform):
        torch, _, _ = require_ml()
        self.torch = torch
        self.frame = frame.reset_index(drop=True)
        self.dataset_path = dataset_path
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.dataset_path / row["image_path"]) as source:
            image = source.convert("RGB")
        label = self.class_to_index[row["scientific_name"]]
        return self.transform(image), self.torch.tensor(label, dtype=self.torch.long)


def _copy_reference_images(frame: pd.DataFrame, dataset_path: Path, model_path: Path) -> dict[str, str]:
    reference_root = model_path / "references"
    reference_root.mkdir(parents=True, exist_ok=True)
    references: dict[str, str] = {}
    for index, (species, rows) in enumerate(frame.groupby("scientific_name")):
        source_path = dataset_path / rows.iloc[0]["image_path"]
        relative_path = Path("references") / f"species-{index:04d}.jpg"
        with Image.open(source_path) as source:
            source.convert("RGB").save(model_path / relative_path, format="JPEG", quality=90)
        references[species] = relative_path.as_posix()
    return references


def train_from_zip(
    blob: bytes,
    dataset_name: str,
    workspace: Path,
    options: TrainingOptions | None = None,
    progress: ProgressCallback | None = None,
) -> ModelRecord:
    installed = install_dataset(blob, dataset_name, workspace / "datasets")
    return train_installed_dataset(installed, workspace / "models", options, progress)


def train_installed_dataset(
    dataset: InstalledDataset,
    models_root: Path,
    options: TrainingOptions | None = None,
    progress: ProgressCallback | None = None,
) -> ModelRecord:
    options = options or TrainingOptions()
    torch, _, _ = require_ml()
    torch.manual_seed(options.seed)

    frame = pd.read_csv(dataset.manifest_path, dtype=str).fillna("")
    classes = sorted(frame["scientific_name"].unique())
    train_frame = frame[frame["split"] == "train"].copy()
    validation_frame = frame[frame["split"] == "validation"].copy()

    train_dataset = _ButterflyDataset(
        train_frame,
        dataset.path,
        classes,
        image_transform(True, options.image_size),
    )
    validation_dataset = _ButterflyDataset(
        validation_frame,
        dataset.path,
        classes,
        image_transform(False, options.image_size),
    )

    class_to_index = {name: index for index, name in enumerate(classes)}
    class_counts = train_frame["scientific_name"].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[name] for name in train_frame["scientific_name"]]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(classes), options.backbone, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=options.label_smoothing)

    timestamp = datetime.now(timezone.utc)
    slug = re.sub(r"[^a-z0-9]+", "-", dataset.name.lower()).strip("-") or "dataset"
    model_id = (
        f"{slug}-{timestamp.strftime('%Y%m%d-%H%M%S')}-"
        f"{dataset.fingerprint[:6]}{uuid4().hex[:4]}"
    )
    model_path = models_root / model_id
    model_path.mkdir(parents=True, exist_ok=False)

    best_accuracy = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, options.epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        validation_correct = 0
        validation_total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                validation_correct += (logits.argmax(dim=1) == labels).sum().item()
                validation_total += labels.size(0)

        scheduler.step()
        metrics = {
            "train_loss": running_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "validation_accuracy": validation_correct / max(validation_total, 1),
        }
        history.append({"epoch": epoch, **metrics})
        if metrics["validation_accuracy"] > best_accuracy:
            best_accuracy = metrics["validation_accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "backbone": options.backbone,
                    "image_size": options.image_size,
                },
                model_path / "best.pt",
            )
        if progress:
            progress(epoch, options.epochs, metrics)

    pd.DataFrame(history).to_csv(model_path / "history.csv", index=False)
    references = _copy_reference_images(frame, dataset.path, model_path)
    metadata = {
        "model_id": model_id,
        "dataset_name": dataset.name,
        "dataset_fingerprint": dataset.fingerprint,
        "created_at": timestamp.isoformat(),
        "backbone": options.backbone,
        "species": classes,
        "references": references,
        "best_validation_accuracy": best_accuracy,
        "training_options": asdict(options),
        "train_images": len(train_frame),
        "validation_images": len(validation_frame),
        "class_to_index": class_to_index,
    }
    write_model_metadata(model_path, metadata)
    return ModelRecord(
        model_id=model_id,
        dataset_name=dataset.name,
        created_at=timestamp.isoformat(),
        path=model_path,
        backbone=options.backbone,
        species=tuple(classes),
        references=references,
        metadata=metadata,
    )
