"""Lazy ML construction so dataset validation and the GUI can run without PyTorch."""

from __future__ import annotations


DEFAULT_BACKBONE = "maxvit_tiny_tf_224.in1k"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def require_ml():
    try:
        import timm
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            'Training and identification require the ML dependencies. Run: pip install -e ".[ml]"'
        ) from exc
    return torch, timm, transforms


def create_model(class_count: int, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True):
    _, timm, _ = require_ml()
    return timm.create_model(backbone, pretrained=pretrained, num_classes=class_count)


def image_transform(training: bool, image_size: int = IMAGE_SIZE):
    _, _, transforms = require_ml()
    normalization = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
                transforms.ToTensor(),
                normalization,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )

