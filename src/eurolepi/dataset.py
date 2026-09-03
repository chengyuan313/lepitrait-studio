"""PyTorch dataset backed by the explicit EuroLepi manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_index: dict[str, int], transform) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(Path(row["image_path"])) as handle:
            image = handle.convert("RGB")
        tensor = self.transform(image)
        target = self.class_to_index[str(row["scientific_name"])]
        return tensor, target, str(row["image_id"])

