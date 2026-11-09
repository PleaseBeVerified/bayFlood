#!/usr/bin/env python3
"""Test a high-elevation, class-balanced FEMA-label training sample.

This experiment does not train on observed flooding and never replaces the app model.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/dnn-training-table.csv"
SELECTED = ROOT / "data/fema-high-elevation-balanced-training.csv"
REPORT_JSON = ROOT / "reports/fema-high-elevation-baseline.json"
REPORT_MD = ROOT / "reports/fema-high-elevation-baseline.md"

PARAMETERS = {
    "n_estimators": 350,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "reg_lambda": 10,
    "n_jobs": 4,
    "random_state": 42,
}


def vector(row: dict[str, str]) -> list[float]:
    distance = row["nearest_water_distance_m"]
    return [
        float(row["elevation_m"]),
        float(row["current_imperviousness_pct"]),
        float(row["current_100yr_24h_rainfall_mm"]),
        float(row["nearest_water_available"]),
        math.log1p(max(0.0, float(distance))) if distance else math.nan,
    ]


def values(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([vector(row) for row in rows], dtype=np.float32),
        np.asarray([int(row["flood_risk_label"]) for row in rows], dtype=np.int64),
    )


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "rocAuc": float(roc_auc_score(labels, scores)),
        "brierScore": float(brier_score_loss(labels, scores)),
        "confusionMatrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    partitions = {
        name: [row for row in rows if row["spatial_partition"] == name]
        for name in ("training", "validation", "test")
    }
    training = partitions["training"]
    positives = [row for row in training if row["flood_risk_label"] == "1"]
    negatives = [row for row in training if row["flood_risk_label"] == "0"]
    if not positives or len(negatives) < len(positives):
        raise RuntimeError("Training split needs positives and at least as many minimal rows")

    selected_negatives = sorted(
        negatives,
        key=lambda row: (-float(row["elevation_m"]), row["location_id"]),
    )[:len(positives)]
    selected_ids = {row["location_id"] for row in positives + selected_negatives}
    holdout_ids = {row["location_id"] for name in ("validation", "test")
                   for row in partitions[name]}
    if selected_ids & holdout_ids or len(selected_ids) != 2 * len(positives):
        raise RuntimeError("Selected sample overlaps holdout or has duplicate locations")

    # Preserve source-table order after deterministic selection.
    selected_rows = [row for row in training if row["location_id"] in selected_ids]
    with SELECTED.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(selected_rows)

    x_train, y_train = values(selected_rows)
    x_validation, y_validation = values(partitions["validation"])
    x_test, y_test = values(partitions["test"])
    model = XGBClassifier(**PARAMETERS)
    model.fit(x_train, y_train)
    validation_scores = model.predict_proba(x_validation)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 161)
    threshold = float(max(thresholds, key=lambda candidate: (
        balanced_accuracy_score(y_validation, validation_scores >= candidate),
        f1_score(y_validation, validation_scores >= candidate),
    )))
    test_scores = model.predict_proba(x_test)[:, 1]

    cutoff = float(selected_negatives[-1]["elevation_m"])
    negative_mask = y_test == 0
    low_mask = negative_mask & (x_test[:, 0] < cutoff)
    high_mask = negative_mask & (x_test[:, 0] >= cutoff)
    report = {
        "purpose": "FEMA-label sampling experiment; not an observed-flood model",
        "input": str(INPUT.relative_to(ROOT)),
        "selectedTrainingTable": str(SELECTED.relative_to(ROOT)),
        "trainingRows": len(selected_rows),
        "trainingPositiveRows": len(positives),
        "selectedMinimalRows": len(selected_negatives),
        "minimalElevationCutoffMeters": cutoff,
        "untouchedValidationRows": len(partitions["validation"]),
        "untouchedTestRows": len(partitions["test"]),
        "parameters": PARAMETERS,
        "decisionThreshold": threshold,
        "validationMetrics": metrics(y_validation, validation_scores, threshold),
        "testMetrics": metrics(y_test, test_scores, threshold),
        "testMinimalFalsePositiveRateBelowTrainingCutoff": float(np.mean(test_scores[low_mask] >= threshold)),
        "testMinimalRowsBelowTrainingCutoff": int(low_mask.sum()),
        "testMinimalFalsePositiveRateAtOrAboveTrainingCutoff": float(np.mean(test_scores[high_mask] >= threshold)),
        "testMinimalRowsAtOrAboveTrainingCutoff": int(high_mask.sum()),
    }
    previous = json.loads((ROOT / "reports/boosted-dnn-comparison.json").read_text())
    report["currentAppTestMetrics"] = previous["testMetrics"]
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    current = report["currentAppTestMetrics"]
    experiment = report["testMetrics"]
    REPORT_MD.write_text(
        "# High-elevation FEMA-label sampling experiment\n\n"
        f"Selected all {len(positives)} training rows mapped to the 1% class and "
        f"the {len(selected_negatives)} highest-elevation training rows mapped "
        f"minimal. The selected minimal cutoff is {cutoff:.1f} m. "
        "Validation and test geographic partitions retain their original class mix. "
        "The threshold was selected on validation only.\n\n"
        "| Geographic test metric | Current app model | High-elevation sample |\n"
        "| --- | ---: | ---: |\n"
        f"| Accuracy | {current['accuracy']:.3f} | {experiment['accuracy']:.3f} |\n"
        f"| Balanced accuracy | {current['balancedAccuracy']:.3f} | {experiment['balancedAccuracy']:.3f} |\n"
        f"| Precision | {current['precision']:.3f} | {experiment['precision']:.3f} |\n"
        f"| Recall | {current['recall']:.3f} | {experiment['recall']:.3f} |\n"
        f"| F1 | {current['f1']:.3f} | {experiment['f1']:.3f} |\n"
        f"| Brier score | {current['brierScore']:.3f} | {experiment['brierScore']:.3f} |\n\n"
        f"On test minimal rows below {cutoff:.1f} m, the false-positive rate is "
        f"{report['testMinimalFalsePositiveRateBelowTrainingCutoff']:.3f} "
        f"({report['testMinimalRowsBelowTrainingCutoff']} rows). At or above the "
        f"cutoff, it is {report['testMinimalFalsePositiveRateAtOrAboveTrainingCutoff']:.3f} "
        f"({report['testMinimalRowsAtOrAboveTrainingCutoff']} rows).\n\n"
        "These labels are FEMA mapped classes, not verified storm flooding. "
        "This experiment must not be presented as event-conditioned flood probability.\n",
        encoding="utf-8",
    )
    print(f"Selected {len(positives)} positive and {len(selected_negatives)} high-elevation minimal training rows")
    print(f"Geographic test balanced accuracy: {experiment['balancedAccuracy']:.3f}")


if __name__ == "__main__":
    main()
