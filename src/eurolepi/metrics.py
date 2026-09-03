"""Metrics kept independent of scikit-learn for portable evaluation."""

from __future__ import annotations

import numpy as np


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("Expected logits [N, C] and targets [N].")
    if len(targets) == 0:
        return 0.0
    k = min(k, logits.shape[1])
    top = np.argpartition(logits, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == targets[:, None], axis=1)))


def macro_f1(predictions: np.ndarray, targets: np.ndarray, class_count: int) -> float:
    scores: list[float] = []
    for class_id in range(class_count):
        true_positive = int(np.sum((predictions == class_id) & (targets == class_id)))
        false_positive = int(np.sum((predictions == class_id) & (targets != class_id)))
        false_negative = int(np.sum((predictions != class_id) & (targets == class_id)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def should_reject(probabilities: np.ndarray, threshold: float) -> bool:
    if probabilities.ndim != 1 or probabilities.size == 0:
        return True
    return bool(float(np.max(probabilities)) < threshold)

