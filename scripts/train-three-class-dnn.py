#!/usr/bin/env python3
"""Train a three-class FEMA-equivalent screening model with geographic holdouts."""

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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "dnn-three-class-training-table.csv"
MODEL_PATH = ROOT / "models" / "flood-hazard-three-class-dnn.joblib"
MODEL_JSON_PATH = ROOT / "models" / "flood-hazard-three-class-dnn.json"
REPORT_JSON_PATH = ROOT / "reports" / "three-class-dnn-evaluation.json"
REPORT_MARKDOWN_PATH = ROOT / "reports" / "three-class-dnn-evaluation.md"

SEED = 42
CLASSES = ["minimal", "one_percent", "point_two_percent"]
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
        "spatial_partition", "elevation_m", "current_imperviousness_pct",
        "current_100yr_24h_rainfall_mm", "nearest_water_available",
        "nearest_water_distance_m", "mapped_hazard_class", "county",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(sorted(missing))}")
    unknown = {row["mapped_hazard_class"] for row in rows}.difference(CLASSES)
    if unknown:
        raise RuntimeError(f"Unknown hazard classes: {', '.join(sorted(unknown))}")
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


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(CLASSES)[np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=CLASSES, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balancedAccuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macroF1": float(f1_score(y_true, predictions, labels=CLASSES, average="macro")),
        "logLoss": float(log_loss(y_true, probabilities, labels=CLASSES)),
        "perClass": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASSES)
        },
        "confusionMatrix": confusion_matrix(y_true, predictions, labels=CLASSES).tolist(),
    }


def exported_probabilities(model: MLPClassifier, values: np.ndarray) -> np.ndarray:
    activations = values
    for weights, biases in zip(model.coefs_[:-1], model.intercepts_[:-1], strict=True):
        activations = np.maximum(0.0, activations @ weights + biases)
    logits = activations @ model.coefs_[-1] + model.intercepts_[-1]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def markdown_report(report: dict[str, Any]) -> str:
    test = report["testMetrics"]
    newark = report["newarkTestAudit"]
    per_class = "\n".join(
        f"| {name} | {values['support']} | {values['precision']:.3f} | "
        f"{values['recall']:.3f} | {values['f1']:.3f} |"
        for name, values in test["perClass"].items()
    )
    return f"""# Three-Class DNN Evaluation

This candidate predicts FEMA-equivalent mapped hazard categories from five
physical inputs. The 0.2%-annual-chance category is included. Training and
selection used geographically separated training and validation blocks;
the test blocks were evaluated only after selecting the model.

## Data

- Training rows: {report['rowCounts']['training']}
- Validation rows: {report['rowCounts']['validation']}
- Test rows: {report['rowCounts']['test']}
- Hidden layers: {report['selectedCandidate']['hiddenLayerSizes']}
- Features: {', '.join(report['features'])}

## Held-Out Test Results

- Accuracy: {test['accuracy']:.3f}
- Balanced accuracy: {test['balancedAccuracy']:.3f}
- Macro F1: {test['macroF1']:.3f}
- Log loss: {test['logLoss']:.3f}
- Majority-class baseline accuracy: {report['majorityClassTestAccuracy']:.3f}

| FEMA-equivalent class | Test rows | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
{per_class}

Confusion matrix (rows are actual, columns predicted; class order:
`{', '.join(CLASSES)}`): `{test['confusionMatrix']}`.

## Newark Spot Check

Within 3 km of Newark City Hall, {newark['testRows']} held-out test rows were
available. The model correctly classified {newark['correct']} of them;
{newark['minimalCalledModerate']} mapped minimal-hazard rows were predicted
as the 0.2%-annual-chance class. This post-hoc diagnostic was not used to
select the model. It shows why this candidate is not ready to fill map gaps.

These are results for predicting FEMA map classes in held-out Bay Area blocks.
They do not validate performance in genuinely unmapped areas, and class
probabilities are not the chance that a property will flood. The model omits
local drainage, levees, and building floor elevation. Review errors and
uncertainty before displaying this candidate in the app.
"""


def main() -> None:
    rows = load_rows()
    partitions = {
        name: [row for row in rows if row["spatial_partition"] == name]
        for name in ("training", "validation", "test")
    }
    if any(not partition for partition in partitions.values()):
        raise RuntimeError("Training, validation, and test partitions must all be nonempty")
    if any(set(CLASSES) != {row["mapped_hazard_class"] for row in values} for values in partitions.values()):
        raise RuntimeError("Every partition must include all three mapped classes")

    x_train_raw = vectorize(partitions["training"])
    x_validation_raw = vectorize(partitions["validation"])
    x_test_raw = vectorize(partitions["test"])
    y_train = np.asarray([row["mapped_hazard_class"] for row in partitions["training"]])
    y_validation = np.asarray([row["mapped_hazard_class"] for row in partitions["validation"]])
    y_test = np.asarray([row["mapped_hazard_class"] for row in partitions["test"]])

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw))
    x_validation = scaler.transform(imputer.transform(x_validation_raw))
    x_test = scaler.transform(imputer.transform(x_test_raw))
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    candidates = []
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
        validation_probabilities = model.predict_proba(x_validation)
        validation_metrics = metrics(y_validation, validation_probabilities)
        result = {
            "hiddenLayerSizes": list(candidate["hidden_layer_sizes"]),
            "alpha": candidate["alpha"],
            "iterations": int(model.n_iter_),
            "validationMacroF1": validation_metrics["macroF1"],
            "validationBalancedAccuracy": validation_metrics["balancedAccuracy"],
        }
        candidates.append(result)
        trained_models.append(model)
        print(f"Candidate {result['hiddenLayerSizes']}: validation macro F1={result['validationMacroF1']:.4f}")

    selected_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["validationMacroF1"],
            candidates[index]["validationBalancedAccuracy"],
        ),
    )
    model = trained_models[selected_index]
    if list(model.classes_) != CLASSES or model.out_activation_ != "softmax":
        raise RuntimeError("Unexpected three-class output order or activation")
    validation_metrics = metrics(y_validation, model.predict_proba(x_validation))
    test_probabilities = model.predict_proba(x_test)
    test_metrics = metrics(y_test, test_probabilities)
    test_predictions = model.predict(x_test)
    newark_latitude, newark_longitude = 37.53589, -122.02951
    newark_indices = [
        index for index, row in enumerate(partitions["test"])
        if 111.0 * math.hypot(
            float(row["lat"]) - newark_latitude,
            math.cos(math.radians(newark_latitude)) * (float(row["lng"]) - newark_longitude),
        ) <= 3.0
    ]
    newark_audit = {
        "center": {"latitude": newark_latitude, "longitude": newark_longitude},
        "radiusKm": 3,
        "testRows": len(newark_indices),
        "correct": int(sum(y_test[index] == test_predictions[index] for index in newark_indices)),
        "minimalCalledModerate": int(sum(
            y_test[index] == "minimal" and test_predictions[index] == "point_two_percent"
            for index in newark_indices
        )),
    }
    parity_error = float(np.max(np.abs(exported_probabilities(model, x_test) - test_probabilities)))
    if parity_error > 1e-10:
        raise RuntimeError(f"JSON export parity failed: maximum error {parity_error}")

    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT_PATH.relative_to(ROOT)),
        "target": "FEMA-equivalent mapped class: minimal, point_two_percent, one_percent",
        "features": FEATURES,
        "classes": CLASSES,
        "rowCounts": {name: len(values) for name, values in partitions.items()},
        "labelCounts": {
            name: dict(sorted(Counter(row["mapped_hazard_class"] for row in values).items()))
            for name, values in partitions.items()
        },
        "candidates": candidates,
        "selectedCandidate": candidates[selected_index],
        "validationMetrics": validation_metrics,
        "testMetrics": test_metrics,
        "newarkTestAudit": newark_audit,
        "majorityClassTestAccuracy": float(max(np.mean(y_test == label) for label in CLASSES)),
        "jsonExportMaximumProbabilityError": parity_error,
    }
    payload = {
        "schemaVersion": 1,
        "modelType": "feedforward-neural-network-three-class-screening",
        "generatedAt": report["generatedAt"],
        "features": FEATURES,
        "classes": CLASSES,
        "preprocessing": {
            "medianImputation": imputer.statistics_.tolist(),
            "standardizationMean": scaler.mean_.tolist(),
            "standardizationScale": scaler.scale_.tolist(),
        },
        "hiddenActivation": "relu",
        "outputActivation": "softmax",
        "layers": [
            {"weights": weights.tolist(), "biases": biases.tolist()}
            for weights, biases in zip(model.coefs_, model.intercepts_, strict=True)
        ],
        "evaluation": test_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "imputer": imputer, "scaler": scaler, "features": FEATURES}, MODEL_PATH)
    MODEL_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT_MARKDOWN_PATH.write_text(markdown_report(report), encoding="utf-8")
    print(f"Held-out test: accuracy={test_metrics['accuracy']:.4f}, balanced accuracy={test_metrics['balancedAccuracy']:.4f}, macro F1={test_metrics['macroF1']:.4f}")


if __name__ == "__main__":
    main()
