#!/usr/bin/env python3
"""Fit the buffered XGBoost comparator in a process without PyTorch loaded."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from terrain_features import joined_rows


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/terrain-cnn"


def main() -> None:
    rows = json.loads((CACHE / "metadata.json").read_text())["rows"]
    table, vectors = joined_rows(ROOT / "data/dnn-training-table.csv",
                                 ROOT / "data/physical-hazard-calibration.csv")
    if [r["location_id"] for r in rows] != [r["location_id"] for r in table]:
        raise RuntimeError("Patch cache order differs from tabular rows")
    xy = np.asarray([[r["x_m"], r["y_m"]] for r in rows])
    splits = {name: np.asarray([i for i, r in enumerate(rows) if r["partition"] == name])
              for name in ("training", "validation", "test")}
    test_tree = cKDTree(xy[splits["test"]])
    validation = splits["validation"]
    validation = validation[np.isinf(test_tree.query(xy[validation], p=np.inf,
                                                   distance_upper_bound=1920)[0])]
    holdout = cKDTree(xy[np.concatenate((validation, splits["test"]))])
    training = splits["training"]
    training = training[np.isinf(holdout.query(xy[training], p=np.inf,
                                             distance_upper_bound=1920)[0])]
    x = np.asarray([vectors[r["location_id"]] for r in rows], dtype=np.float32)
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    model = XGBClassifier(n_estimators=350, max_depth=3, learning_rate=0.03,
                          subsample=0.85, colsample_bytree=0.9, reg_lambda=10,
                          n_jobs=4, random_state=42)
    model.fit(x[training], y[training])
    validation_scores = model.predict_proba(x[validation])[:, 1]
    positive = np.sort(validation_scores[y[validation] == 1])
    threshold = float(positive[len(positive) - math.ceil(0.95 * len(positive))])
    test_scores = model.predict_proba(x[splits["test"]])[:, 1]

    def metrics(indices: np.ndarray, scores: np.ndarray) -> dict:
        labels = y[indices]
        predicted = scores >= threshold
        negative = labels == 0
        return {
            "rows": len(labels), "minimalRows": int(negative.sum()),
            "onePercentRows": int((labels == 1).sum()),
            "accuracy": float(accuracy_score(labels, predicted)),
            "balancedAccuracy": float(balanced_accuracy_score(labels, predicted)),
            "minimalFalsePositiveRate": float(predicted[negative].mean()),
            "minimalFalsePositives": int(predicted[negative].sum()),
            "onePercentRecall": float(recall_score(labels, predicted)),
            "onePercentPrecision": float(precision_score(labels, predicted, zero_division=0)),
            "rocAuc": float(roc_auc_score(labels, scores)),
            "brierScore": float(brier_score_loss(labels, scores)),
            "confusionMatrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        }

    (CACHE / "baseline.json").write_text(json.dumps({
        "trainingIndices": training.tolist(), "validationIndices": validation.tolist(),
        "testIndices": splits["test"].tolist(), "validationThreshold": threshold,
        "validationScores": validation_scores.tolist(), "testScores": test_scores.tolist(),
        "validation": metrics(validation, validation_scores),
        "test": metrics(splits["test"], test_scores),
    }), encoding="utf-8")
    print(f"Buffered tabular baseline: test FPR={metrics(splits['test'], test_scores)['minimalFalsePositiveRate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
