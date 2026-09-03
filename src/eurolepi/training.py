"""Reproducible MaxViT-T training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from .manifest import validate_manifest
from .metrics import macro_f1, topk_accuracy
from .modeling import DEFAULT_BACKBONE, create_classifier, create_transform, load_torch_checkpoint


@dataclass
class TrainingConfig:
    manifest: str = "data/manifest.csv"
    output_dir: str = "models/eurolepi_maxvit_tiny"
    backbone: str = DEFAULT_BACKBONE
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    balanced_sampling: bool = True
    num_workers: int = 0
    seed: int = 42
    patience: int = 6
    reject_threshold: float = 0.65
    pretrained: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install training dependencies with: pip install -e .[ml]") from exc
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown training config fields: {sorted(unknown)}")
        return cls(**values)


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loader(frame, class_to_index, transform, config, torch, *, training: bool):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from .dataset import ButterflyDataset

    dataset = ButterflyDataset(frame, class_to_index, transform)
    sampler = None
    shuffle = training
    if training and config.balanced_sampling:
        counts = frame["scientific_name"].value_counts()
        weights = frame["scientific_name"].map(lambda name: 1.0 / counts[name]).to_numpy()
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(model, loader, criterion, device, torch, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().float().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())
    logits_np = np.concatenate(logits_all) if logits_all else np.empty((0, 0))
    targets_np = np.concatenate(targets_all) if targets_all else np.empty(0, dtype=int)
    predictions = logits_np.argmax(axis=1) if len(logits_np) else np.empty(0, dtype=int)
    class_count = logits_np.shape[1] if logits_np.ndim == 2 else 0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1": topk_accuracy(logits_np, targets_np, 1) if len(targets_np) else 0.0,
        "top5": topk_accuracy(logits_np, targets_np, 5) if len(targets_np) else 0.0,
        "macro_f1": macro_f1(predictions, targets_np, class_count) if class_count else 0.0,
    }


def train(config: TrainingConfig) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e .[ml]") from exc

    frame = pd.read_csv(config.manifest)
    report = validate_manifest(frame, require_files=True, require_split=True)
    if not report.valid:
        raise ValueError("Invalid training manifest: " + "; ".join(report.errors))
    _seed_everything(config.seed, torch)

    classes = sorted(frame.loc[frame["split"] == "train", "scientific_name"].unique())
    class_to_index = {name: index for index, name in enumerate(classes)}
    train_frame = frame[frame["split"] == "train"]
    validation_frame = frame[frame["split"] == "validation"]
    if validation_frame.empty:
        raise ValueError("Validation split is empty.")

    train_loader = _build_loader(
        train_frame,
        class_to_index,
        create_transform(image_size=config.image_size, training=True),
        config,
        torch,
        training=True,
    )
    validation_loader = _build_loader(
        validation_frame,
        class_to_index,
        create_transform(image_size=config.image_size, training=False),
        config,
        torch,
        training=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_classifier(
        len(classes), backbone=config.backbone, pretrained=config.pretrained
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    stale_epochs = 0
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, torch, optimizer=optimizer, scaler=scaler
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model, validation_loader, criterion, device, torch
            )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "backbone": config.backbone,
                    "image_size": config.image_size,
                    "reject_threshold": config.reject_threshold,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "config": asdict(config),
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "class_names.json").write_text(
        json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    model_card = {
        "task": "European butterfly species identification",
        "backbone": config.backbone,
        "classes": len(classes),
        "training_images": len(train_frame),
        "validation_images": len(validation_frame),
        "best_validation_macro_f1": best_f1,
        "domains": sorted(frame["domain"].unique().tolist()),
        "limitations": [
            "Predictions outside the training species are not taxonomic determinations.",
            "Museum and in-situ field domains must be evaluated separately.",
            "Low-confidence results require expert review.",
        ],
    }
    (output_dir / "model_card.json").write_text(
        json.dumps(model_card, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return checkpoint_path


def evaluate(checkpoint_path: str | Path, manifest_path: str | Path, split: str = "test"):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e .[ml]") from exc
    checkpoint = load_torch_checkpoint(checkpoint_path)
    classes = list(checkpoint["classes"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    frame = pd.read_csv(manifest_path)
    frame = frame[frame["split"] == split]
    unknown = sorted(set(frame["scientific_name"]) - set(classes))
    if unknown:
        raise ValueError(f"Evaluation contains classes absent from checkpoint: {unknown[:10]}")
    config = TrainingConfig(
        manifest=str(manifest_path),
        backbone=checkpoint["backbone"],
        image_size=int(checkpoint["image_size"]),
        pretrained=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_classifier(
        len(classes), backbone=checkpoint["backbone"], pretrained=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    loader = _build_loader(
        frame,
        class_to_index,
        create_transform(image_size=config.image_size, training=False),
        config,
        torch,
        training=False,
    )
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        return _run_epoch(model, loader, criterion, device, torch)

