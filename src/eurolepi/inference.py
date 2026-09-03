"""Checkpoint-backed inference with an explicit low-confidence rejection decision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .modeling import create_classifier, create_transform, load_torch_checkpoint
from .schemas import IdentificationResult, Prediction


class ButterflyIdentifier:
    def __init__(self, checkpoint_path: str | Path, *, threshold: float | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install inference dependencies with: pip install -e .[ml]") from exc
        self.torch = torch
        self.path = Path(checkpoint_path)
        checkpoint = load_torch_checkpoint(self.path)
        self.classes = list(checkpoint["classes"])
        self.backbone = str(checkpoint["backbone"])
        self.image_size = int(checkpoint.get("image_size", 224))
        self.threshold = float(
            checkpoint.get("reject_threshold", 0.65) if threshold is None else threshold
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_classifier(
            len(self.classes), backbone=self.backbone, pretrained=False
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.transform = create_transform(image_size=self.image_size, training=False)

    def predict(self, image: Image.Image, *, top_k: int = 5) -> IdentificationResult:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(tensor)[0]
            probabilities = self.torch.softmax(logits, dim=0).cpu().numpy()
        top_k = min(top_k, len(self.classes))
        indices = np.argsort(probabilities)[::-1][:top_k]
        predictions = tuple(
            Prediction(
                scientific_name=self.classes[int(index)],
                probability=float(probabilities[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        )
        rejected = not predictions or predictions[0].probability < self.threshold
        return IdentificationResult(
            predictions=predictions,
            rejected=rejected,
            threshold=self.threshold,
            model_name=self.backbone,
        )

