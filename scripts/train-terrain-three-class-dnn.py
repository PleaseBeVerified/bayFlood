#!/usr/bin/env python3
"""Train a ten-input, three-class DNN on FEMA mapped hazard categories."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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

from terrain_features import FEATURES, joined_rows


ROOT = Path(__file__).resolve().parents[1]
TRAINING_TABLE = ROOT / "data/dnn-three-class-training-table.csv"
CALIBRATION = ROOT / "data/physical-hazard-calibration.csv"
MODEL_PATH = ROOT / "models/flood-hazard-terrain-three-class-dnn.joblib"
ARTIFACT_PATH = ROOT / "models/flood-hazard-terrain-three-class-dnn.json"
REPORT_JSON = ROOT / "reports/terrain-three-class-dnn-evaluation.json"
REPORT_MD = ROOT / "reports/terrain-three-class-dnn-evaluation.md"
CLASSES = ["minimal", "one_percent", "point_two_percent"]
CANDIDATES = [
    {"hidden_layer_sizes": (32, 16), "alpha": 0.001, "class_weight": "balanced"},
    {"hidden_layer_sizes": (64, 32, 16), "alpha": 0.001, "class_weight": "balanced"},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.01, "class_weight": "balanced"},
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.01, "class_weight": "balanced"},
    {"hidden_layer_sizes": (32, 32, 16), "alpha": 0.01, "class_weight": "balanced"},
    {"hidden_layer_sizes": (64, 32, 16), "alpha": 0.1, "class_weight": "balanced"},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.01, "class_weight": "none"},
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.01, "class_weight": "none"},
]


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = np.asarray(CLASSES)[np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=CLASSES, zero_division=0,
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


def predict_exported(raw: np.ndarray, artifact: dict) -> np.ndarray:
    preprocessing = artifact["preprocessing"]
    median = np.asarray(preprocessing["medianImputation"])
    mean = np.asarray(preprocessing["standardizationMean"])
    scale = np.asarray(preprocessing["standardizationScale"])
    activations = (np.where(np.isnan(raw), median, raw) - mean) / scale
    for layer in artifact["layers"][:-1]:
        activations = np.maximum(0, activations @ np.asarray(layer["weights"]) + layer["biases"])
    output = artifact["layers"][-1]
    logits = activations @ np.asarray(output["weights"]) + output["biases"]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def main() -> None:
    rows, x_by_id = joined_rows(TRAINING_TABLE, CALIBRATION, "mapped_hazard_class")
    partitions = {
        name: [row for row in rows if row["spatial_partition"] == name]
        for name in ("training", "validation", "test")
    }
    if any({row["mapped_hazard_class"] for row in values} != set(CLASSES)
           for values in partitions.values()):
        raise RuntimeError("Every geographic partition must contain all three classes")
    raw = {
        name: np.asarray([x_by_id[row["location_id"]] for row in values], dtype=np.float64)
        for name, values in partitions.items()
    }
    labels = {
        name: np.asarray([row["mapped_hazard_class"] for row in values])
        for name, values in partitions.items()
    }
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(raw["training"]))
    x_validation = scaler.transform(imputer.transform(raw["validation"]))
    x_test = scaler.transform(imputer.transform(raw["test"]))
    balanced_weights = compute_sample_weight(class_weight="balanced", y=labels["training"])

    candidate_results = []
    models = []
    for candidate in CANDIDATES:
        model = MLPClassifier(
            hidden_layer_sizes=candidate["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=candidate["alpha"],
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
        )
        weights = balanced_weights if candidate["class_weight"] == "balanced" else None
        model.fit(x_train, labels["training"], sample_weight=weights)
        validation = metrics(labels["validation"], model.predict_proba(x_validation))
        candidate_results.append({
            "hiddenLayerSizes": list(candidate["hidden_layer_sizes"]),
            "alpha": candidate["alpha"],
            "classWeight": candidate["class_weight"],
            "iterations": int(model.n_iter_),
            "validationMacroF1": validation["macroF1"],
            "validationBalancedAccuracy": validation["balancedAccuracy"],
            "validationPointTwoF1": validation["perClass"]["point_two_percent"]["f1"],
        })
        models.append(model)
        print(f"Candidate {candidate}: validation macro F1 "
              f"{validation['macroF1']:.3f}, 0.2% F1 "
              f"{validation['perClass']['point_two_percent']['f1']:.3f}", flush=True)

    selected = max(range(len(models)), key=lambda index: (
        candidate_results[index]["validationMacroF1"],
        candidate_results[index]["validationBalancedAccuracy"],
    ))
    model = models[selected]
    if list(model.classes_) != CLASSES or model.out_activation_ != "softmax":
        raise RuntimeError("Unexpected class order or output activation")
    validation_metrics = metrics(labels["validation"], model.predict_proba(x_validation))
    test_probabilities = model.predict_proba(x_test)
    test_metrics = metrics(labels["test"], test_probabilities)
    artifact = {
        "schemaVersion": 1,
        "modelType": "feedforward-neural-network-three-class-screening",
        "target": "FEMA mapped class; not observed flooding",
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
            {"weights": coefficient.tolist(), "biases": intercept.tolist()}
            for coefficient, intercept in zip(model.coefs_, model.intercepts_, strict=True)
        ],
    }
    parity_error = float(np.max(np.abs(predict_exported(raw["test"], artifact) - test_probabilities)))
    if parity_error > 1e-10:
        raise RuntimeError(f"Export parity error: {parity_error}")
    reference_inputs = raw["test"][:5]
    reference_scores = predict_exported(reference_inputs, artifact)
    old = json.loads((ROOT / "reports/three-class-dnn-evaluation.json").read_text())
    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "target": artifact["target"],
        "inputFiles": [str(TRAINING_TABLE.relative_to(ROOT)), str(CALIBRATION.relative_to(ROOT))],
        "features": FEATURES,
        "classes": CLASSES,
        "partitionCounts": {name: len(values) for name, values in partitions.items()},
        "labelCounts": {
            name: {label: int((labels[name] == label).sum()) for label in CLASSES}
            for name in partitions
        },
        "candidates": candidate_results,
        "selectedCandidate": candidate_results[selected],
        "selectionCriterion": "highest geographic-validation macro F1, then balanced accuracy",
        "validationMetrics": validation_metrics,
        "testMetrics": test_metrics,
        "fiveInputTestMetrics": old["testMetrics"],
        "exportMaximumProbabilityError": parity_error,
        "referenceCases": [
            {"inputs": [None if np.isnan(value) else float(value) for value in values],
             "probabilities": [float(value) for value in scores]}
            for values, scores in zip(reference_inputs, reference_scores, strict=True)
        ],
    }
    joblib.dump({"model": model, "imputer": imputer, "scaler": scaler,
                 "features": FEATURES, "classes": CLASSES}, MODEL_PATH)
    ARTIFACT_PATH.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Three-class terrain DNN evaluation", "",
        "All 10,000 FEMA-labeled samples are used, including 933 mapped 0.2% "
        "annual-chance locations. The model predicts mapped minimal, 0.2%, or 1% "
        "hazard categories from the same ten features as the terrain binary models. "
        "Eight candidates were selected by geographic-validation macro F1. "
        "The app continues to serve the binary terrain boosted score.", "",
        "| Geographic test metric | Five-input three-class DNN | Ten-input three-class DNN |",
        "| --- | ---: | ---: |",
    ]
    for label, key in [
        ("Accuracy", "accuracy"), ("Balanced accuracy", "balancedAccuracy"),
        ("Macro F1", "macroF1"), ("Log loss", "logLoss"),
    ]:
        lines.append(f"| {label} | {old['testMetrics'][key]:.3f} | {test_metrics[key]:.3f} |")
    lines += ["", "| Ten-input test class | Samples | Precision | Recall | F1 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in CLASSES:
        result = test_metrics["perClass"][name]
        lines.append(f"| {name} | {result['support']} | {result['precision']:.3f} | "
                     f"{result['recall']:.3f} | {result['f1']:.3f} |")
    lines += ["", "Confusion matrix (rows observed, columns predicted; class order "
              f"{', '.join(CLASSES)}): `{test_metrics['confusionMatrix']}`.", "",
              "These are FEMA map-class labels, not observed floods. An unmapped "
              "address has no independent label in this evaluation, and the class "
              "scores are not annual flooding probabilities.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Test macro F1 {test_metrics['macroF1']:.3f}; "
          f"0.2% precision {test_metrics['perClass']['point_two_percent']['precision']:.3f}; "
          f"0.2% recall {test_metrics['perClass']['point_two_percent']['recall']:.3f}")


if __name__ == "__main__":
    main()
