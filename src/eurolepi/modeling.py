"""Model and image-transform factories.

The default name matches timm's ImageNet-1K MaxViT-T checkpoint and mirrors the
architecture selected by Barkmann et al. (2026).
"""

from __future__ import annotations


DEFAULT_BACKBONE = "maxvit_tiny_tf_224.in1k"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def require_ml_dependencies():
    try:
        import timm
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "Machine-learning dependencies are not installed. Run: pip install -e .[ml]"
        ) from exc
    return timm, torch, torchvision


def create_classifier(
    class_count: int,
    *,
    backbone: str = DEFAULT_BACKBONE,
    pretrained: bool = True,
    drop_rate: float = 0.1,
):
    timm, _, _ = require_ml_dependencies()
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=class_count,
        drop_rate=drop_rate,
    )


def create_transform(*, image_size: int = 224, training: bool = False):
    _, _, torchvision = require_ml_dependencies()
    transforms = torchvision.transforms
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_torch_checkpoint(path, *, map_location="cpu"):
    _, torch, _ = require_ml_dependencies()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

