#!/usr/bin/env python3
"""Train and evaluate a tabular deep neural network for flood-risk screening."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "dnn-training-table.csv"
MODEL_PATH = ROOT / "models" / "flood-risk-dnn.joblib"
MODEL_JSON_PATH = ROOT / "models" / "flood-risk-dnn.json"
REPORT_JSON_PATH = ROOT / "reports" / "dnn-model-evaluation.json"
REPORT_MARKDOWN_PATH = ROOT / "reports" / "dnn-model-evaluation.md"

SEED = 42
FEATURES = [
    "elevation_m",
    "current_imperviousness_pct",
    "current_100yr_24h_rainfall_mm",
    "nearest_water_available",
    "log_nearest_water_distance_m",
]
CANDIDATES = [
    {"hidden_layer_sizes": (32, 16), "alpha": 0.001},
    {"hidden_layer_sizes": (64, 32, 16), "alpha": 0.001},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.01},
]


def load_rows() -> list[dict[str, str]]:
    with INPUT_PATH.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) < 1_000:
        raise RuntimeError(f"{INPUT_PATH} needs at least 1,000 rows")
    required = {
        "spatial_partition",
        "elevation_m",
        "current_imperviousness_pct",
        "current_100yr_24h_rainfall_mm",
        "nearest_water_available",
        "nearest_water_distance_m",
        "flood_risk_label",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(sorted(missing))}")
    return rows


def optional_number(value: str | None) -> float:
    if value is None or value.strip() == "":
        return math.nan
    number = float(value)
    return number if math.isfinite(number) else math.nan


def vectorize(rows: list[dict[str, str]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    result = np.empty((len(rows), len(FEATURES)), dtype=np.float64)
    for index, row in enumerate(rows):
        distance = optional_number(row.get("nearest_water_distance_m"))
        result[index] = [
            optional_number(row.get("elevation_m")),
            optional_number(row.get("current_imperviousness_pct")),
            optional_number(row.get("current_100yr_24h_rainfall_mm")),
            optional_number(row.get("nearest_water_available")),
            math.log1p(max(0.0, distance)) if math.isfinite(distance) else math.nan,
        ]
    return result


def labels(rows: list[dict[str, str]]) -> np.ndarray[Any, np.dtype[np.int64]]:
    return np.asarray([int(row["flood_risk_label"]) for row in rows], dtype=np.int64)


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balancedAccuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "rocAuc": float(roc_auc_score(y_true, probabilities)),
        "averagePrecision": float(average_precision_score(y_true, probabilities)),
        "brierScore": float(brier_score_loss(y_true, probabilities)),
        "confusionMatrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.10, 0.90, 161)
    return float(
        max(
            candidates,
            key=lambda threshold: (
                balanced_accuracy_score(y_true, probabilities >= threshold),
                f1_score(y_true, probabilities >= threshold, zero_division=0),
            ),
        )
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -709, 709)))


def exported_probabilities(model: MLPClassifier, values: np.ndarray) -> np.ndarray:
    activations = values
    for weights, biases in zip(model.coefs_[:-1], model.intercepts_[:-1], strict=True):
        activations = np.maximum(0.0, activations @ weights + biases)
    return sigmoid(activations @ model.coefs_[-1] + model.intercepts_[-1]).reshape(-1)


def export_json(
    model: MLPClassifier,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    threshold: float,
    report: dict[str, Any],
) -> None:
    payload = {
        "schemaVersion": 1,
        "modelType": "feedforward-neural-network-binary-classifier",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "preprocessing": {
            "medianImputation": imputer.statistics_.tolist(),
            "standardizationMean": scaler.mean_.tolist(),
            "standardizationScale": scaler.scale_.tolist(),
        },
        "hiddenActivation": "relu",
        "outputActivation": "sigmoid",
        "decisionThreshold": threshold,
        "layers": [
            {"weights": weights.tolist(), "biases": biases.tolist()}
            for weights, biases in zip(model.coefs_, model.intercepts_, strict=True)
        ],
        "evaluation": report["testMetrics"],
    }
    MODEL_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    test = report["testMetrics"]
    matrix = test["confusionMatrix"]
    return f"""# DNN Model Evaluation

This report evaluates a tabular feedforward neural network on spatially separated
Bay Area blocks. Architecture selection and threshold tuning used validation data;
the test partition remained untouched until final evaluation.

## Data

- Training rows: {report['rowCounts']['training']}
- Validation rows: {report['rowCounts']['validation']}
- Test rows: {report['rowCounts']['test']}
- Features: {', '.join(report['features'])}
- Hidden layers: {report['selectedCandidate']['hiddenLayerSizes']}
- Decision threshold: {report['decisionThreshold']:.3f}

## Untouched Test Results

- Accuracy: {test['accuracy']:.4f}
- Balanced accuracy: {test['balancedAccuracy']:.4f}
- ROC AUC: {test['rocAuc']:.4f}
- Average precision: {test['averagePrecision']:.4f}
- Precision: {test['precision']:.4f}
- Recall: {test['recall']:.4f}
- F1: {test['f1']:.4f}
- Brier score: {test['brierScore']:.4f}
- Confusion matrix (`[[TN, FP], [FN, TP]]`): `{matrix}`

Accuracy alone is not sufficient because the dataset is imbalanced. Balanced
accuracy, ROC AUC, precision, recall, and calibration should be considered together.
The FEMA-derived target is a screening label, not a literal probability that a
specific property will flood.
"""


def main() -> None:
    rows = load_rows()
    partitions = {
        name: [row for row in rows if row["spatial_partition"] == name]
        for name in ("training", "validation", "test")
    }
    if any(not partition for partition in partitions.values()):
        raise RuntimeError("Training, validation, and test partitions must all be nonempty")

    x_train_raw = vectorize(partitions["training"])
    x_validation_raw = vectorize(partitions["validation"])
    x_test_raw = vectorize(partitions["test"])
    y_train = labels(partitions["training"])
    y_validation = labels(partitions["validation"])
    y_test = labels(partitions["test"])

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw))
    x_validation = scaler.transform(imputer.transform(x_validation_raw))
    x_test = scaler.transform(imputer.transform(x_test_raw))
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    candidate_results = []
    trained_models = []
    for candidate in CANDIDATES:
        model = MLPClassifier(
            hidden_layer_sizes=candidate["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=candidate["alpha"],
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
        )
        model.fit(x_train, y_train, sample_weight=sample_weights)
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        candidate_result = {
            "hiddenLayerSizes": list(candidate["hidden_layer_sizes"]),
            "alpha": candidate["alpha"],
            "iterations": int(model.n_iter_),
            "validationRocAuc": float(roc_auc_score(y_validation, validation_probabilities)),
            "validationAveragePrecision": float(
                average_precision_score(y_validation, validation_probabilities)
            ),
        }
        candidate_results.append(candidate_result)
        trained_models.append(model)
        print(
            f"Candidate {candidate_result['hiddenLayerSizes']}: "
            f"validation ROC AUC={candidate_result['validationRocAuc']:.4f}"
        )

    selected_index = max(
        range(len(candidate_results)),
        key=lambda index: (
            candidate_results[index]["validationRocAuc"],
            candidate_results[index]["validationAveragePrecision"],
        ),
    )
    model = trained_models[selected_index]
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = select_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict_proba(x_test)[:, 1]

    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT_PATH.relative_to(ROOT)),
        "target": "flood_risk_label (FEMA one_percent=1, minimal=0)",
        "features": FEATURES,
        "rowCounts": {name: len(values) for name, values in partitions.items()},
        "labelCounts": {
            name: dict(sorted(Counter(row["flood_risk_label"] for row in values).items()))
            for name, values in partitions.items()
        },
        "candidates": candidate_results,
        "selectedCandidate": candidate_results[selected_index],
        "decisionThreshold": threshold,
        "validationMetrics": metrics(y_validation, validation_probabilities, threshold),
        "testMetrics": metrics(y_test, test_probabilities, threshold),
        "majorityClassTestAccuracy": float(max(np.mean(y_test == 0), np.mean(y_test == 1))),
    }

    exported = exported_probabilities(model, x_test)
    max_parity_error = float(np.max(np.abs(exported - test_probabilities)))
    if max_parity_error > 1e-10:
        raise RuntimeError(f"JSON export parity failed: maximum error {max_parity_error}")
    report["jsonExportMaximumProbabilityError"] = max_parity_error

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "imputer": imputer, "scaler": scaler, "features": FEATURES, "threshold": threshold},
        MODEL_PATH,
    )
    export_json(model, imputer, scaler, threshold, report)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT_MARKDOWN_PATH.write_text(markdown_report(report), encoding="utf-8")

    test = report["testMetrics"]
    print(f"Saved DNN model to {MODEL_PATH}")
    print(
        f"Untouched test: accuracy={test['accuracy']:.4f}, "
        f"balanced_accuracy={test['balancedAccuracy']:.4f}, "
        f"roc_auc={test['rocAuc']:.4f}, f1={test['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
