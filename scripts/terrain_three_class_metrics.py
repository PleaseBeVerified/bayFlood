"""Metrics shared by the three-class CNN and its tabular comparator."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    log_loss, precision_recall_fscore_support,
)


CLASSES = ("minimal", "one_percent", "point_two_percent")


def evaluate(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if probabilities.shape != (len(labels), len(CLASSES)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid three-class probability array")
    predicted = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1, 2], zero_division=0,
    )
    normalized = probabilities.astype(np.float64)
    normalized /= normalized.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predicted)),
        "macroF1": float(f1_score(labels, predicted, labels=[0, 1, 2], average="macro")),
        "logLoss": float(log_loss(labels, normalized, labels=[0, 1, 2])),
        "perClass": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASSES)
        },
        "confusionMatrix": confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist(),
    }
