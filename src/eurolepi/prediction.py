"""Single-image Top-5 identification with local species reference images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from eurolepi.modeling import create_model, image_transform, require_ml
from eurolepi.types import Candidate, IdentificationResult, ModelRecord


class ButterflyIdentifier:
    def __init__(self, record: ModelRecord):
        torch, _, _ = require_ml()
        checkpoint_path = record.path / "best.pt"
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.torch = torch
        self.record = record
        self.classes = list(checkpoint["classes"])
        self.image_size = int(checkpoint.get("image_size", 224))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(
            len(self.classes), checkpoint["backbone"], pretrained=False
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.transform = image_transform(False, self.image_size)

    def predict(self, image: Image.Image, top_k: int = 5) -> IdentificationResult:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(tensor), dim=1)[0]
        count = min(top_k, len(self.classes))
        values, indices = self.torch.topk(probabilities, count)
        candidates: list[Candidate] = []
        for rank, (probability, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            species = self.classes[index]
            reference = self.record.references.get(species)
            reference_path = self.record.path / reference if reference else None
            if reference_path is not None and not reference_path.is_file():
                reference_path = None
            candidates.append(Candidate(rank, species, float(probability), reference_path))
        return IdentificationResult(tuple(candidates))


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")

