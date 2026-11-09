#!/usr/bin/env python3
"""Train a ten-input DNN to classify FEMA mapped one-percent versus minimal areas.

The target is a FEMA map class, not observed flooding during a storm.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

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

from terrain_features import FEATURES, joined_rows


ROOT = Path(__file__).resolve().parents[1]
TRAINING_TABLE = ROOT / "data/dnn-training-table.csv"
CALIBRATION = ROOT / "data/physical-hazard-calibration.csv"
MODEL_PATH = ROOT / "models/flood-risk-terrain-dnn.joblib"
ARTIFACT_PATH = ROOT / "models/flood-risk-terrain-dnn.json"
REPORT_JSON = ROOT / "reports/terrain-dnn-evaluation.json"
REPORT_MD = ROOT / "reports/terrain-dnn-evaluation.md"
SEED = 42
CANDIDATES = [
    {"hidden_layer_sizes": (32, 16), "alpha": 0.001},
    {"hidden_layer_sizes": (64, 32, 16), "alpha": 0.001},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.01},
    {"hidden_layer_sizes": (16, 8), "alpha": 0.01},
    {"hidden_layer_sizes": (32, 16), "alpha": 0.01},
    {"hidden_layer_sizes": (32, 32, 16), "alpha": 0.01},
    {"hidden_layer_sizes": (64, 32), "alpha": 0.01},
    {"hidden_layer_sizes": (64, 32, 16), "alpha": 0.01},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.001},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.1},
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.01},
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.1},
    {"hidden_layer_sizes": (64, 64, 64, 32), "alpha": 0.01},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.01, "learning_rate_init": 0.0003},
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.01, "learning_rate_init": 0.0003},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.01, "batch_size": 256},
    {"hidden_layer_sizes": (64, 64, 32), "alpha": 0.01, "class_weight": "none"},
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.01, "class_weight": "none"},
]


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted)),
        "recall": float(recall_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted)),
        "rocAuc": float(roc_auc_score(labels, scores)),
        "averagePrecision": float(average_precision_score(labels, scores)),
        "brierScore": float(brier_score_loss(labels, scores)),
        "confusionMatrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
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
    return (1 / (1 + np.exp(-np.clip(logits, -709, 709)))).reshape(-1)


def main() -> None:
    rows, x_by_id = joined_rows(TRAINING_TABLE, CALIBRATION)
    partitions = {
        name: [row for row in rows if row["spatial_partition"] == name]
        for name in ("training", "validation", "test")
    }
    if any(not values for values in partitions.values()):
        raise RuntimeError("Training, validation, and test partitions must all be nonempty")
    raw = {
        name: np.asarray([x_by_id[row["location_id"]] for row in values], dtype=np.float64)
        for name, values in partitions.items()
    }
    labels = {
        name: np.asarray([int(row["flood_risk_label"]) for row in values], dtype=np.int64)
        for name, values in partitions.items()
    }
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(raw["training"]))
    x_validation = scaler.transform(imputer.transform(raw["validation"]))
    x_test = scaler.transform(imputer.transform(raw["test"]))
    weights = compute_sample_weight(class_weight="balanced", y=labels["training"])

    candidate_results = []
    models = []
    for candidate in CANDIDATES:
        model = MLPClassifier(
            hidden_layer_sizes=candidate["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=candidate["alpha"],
            batch_size=candidate.get("batch_size", 128),
            learning_rate_init=candidate.get("learning_rate_init", 0.001),
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
        )
        started = perf_counter()
        model.fit(
            x_train,
            labels["training"],
            sample_weight=weights if candidate.get("class_weight", "balanced") == "balanced" else None,
        )
        training_seconds = perf_counter() - started
        scores = model.predict_proba(x_validation)[:, 1]
        candidate_results.append({
            "hiddenLayerSizes": list(candidate["hidden_layer_sizes"]),
            "alpha": candidate["alpha"],
            "batchSize": candidate.get("batch_size", 128),
            "learningRateInit": candidate.get("learning_rate_init", 0.001),
            "classWeight": candidate.get("class_weight", "balanced"),
            "iterations": int(model.n_iter_),
            "trainingSeconds": training_seconds,
            "validationRocAuc": float(roc_auc_score(labels["validation"], scores)),
            "validationAveragePrecision": float(average_precision_score(labels["validation"], scores)),
            "validationBrierScore": float(brier_score_loss(labels["validation"], scores)),
        })
        models.append(model)
        print(f"Candidate {candidate}: {training_seconds:.2f}s training, validation ROC AUC "
              f"{candidate_results[-1]['validationRocAuc']:.4f}", flush=True)

    selected = max(range(len(models)), key=lambda index: (
        candidate_results[index]["validationRocAuc"],
        candidate_results[index]["validationAveragePrecision"],
    ))
    model = models[selected]
    validation_scores = model.predict_proba(x_validation)[:, 1]
    threshold = float(max(np.linspace(0.1, 0.9, 161), key=lambda candidate: (
        balanced_accuracy_score(labels["validation"], validation_scores >= candidate),
        f1_score(labels["validation"], validation_scores >= candidate),
    )))
    test_scores = model.predict_proba(x_test)[:, 1]
    initial_model = models[2]
    initial_validation_scores = initial_model.predict_proba(x_validation)[:, 1]
    initial_threshold = float(max(np.linspace(0.1, 0.9, 161), key=lambda candidate: (
        balanced_accuracy_score(labels["validation"], initial_validation_scores >= candidate),
        f1_score(labels["validation"], initial_validation_scores >= candidate),
    )))
    initial_test_metrics = metrics(
        labels["test"], initial_model.predict_proba(x_test)[:, 1], initial_threshold,
    )
    artifact = {
        "schemaVersion": 1,
        "modelType": "feedforward-neural-network-binary-classifier",
        "target": "FEMA mapped one_percent versus minimal; not observed storm flooding",
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
            {"weights": coefficient.tolist(), "biases": intercept.tolist()}
            for coefficient, intercept in zip(model.coefs_, model.intercepts_, strict=True)
        ],
    }
    parity_error = float(np.max(np.abs(predict_exported(raw["test"], artifact) - test_scores)))
    if parity_error > 1e-10:
        raise RuntimeError(f"DNN export parity error: {parity_error}")
    reference_inputs = np.concatenate((raw["test"][:5], np.asarray(
        [[25, 60, 150, 1, np.nan, 2, 0, 0, np.log1p(500), 5]], dtype=np.float64)))
    reference_scores = predict_exported(reference_inputs, artifact)
    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "target": artifact["target"],
        "inputFiles": [str(TRAINING_TABLE.relative_to(ROOT)), str(CALIBRATION.relative_to(ROOT))],
        "features": FEATURES,
        "partitionCounts": {name: len(values) for name, values in partitions.items()},
        "trainingLabels": {
            "minimal": int((labels["training"] == 0).sum()),
            "one_percent": int((labels["training"] == 1).sum()),
        },
        "candidates": candidate_results,
        "selectedCandidate": candidate_results[selected],
        "selectionCriterion": "highest geographic-validation ROC AUC, then average precision",
        "decisionThreshold": threshold,
        "validationMetrics": metrics(labels["validation"], validation_scores, threshold),
        "testMetrics": metrics(labels["test"], test_scores, threshold),
        "exportMaximumProbabilityError": parity_error,
        "referenceCases": [
            {"inputs": [None if np.isnan(value) else float(value) for value in values],
             "probability": float(score)}
            for values, score in zip(reference_inputs, reference_scores, strict=True)
        ],
    }
    previous = json.loads((ROOT / "reports/dnn-model-evaluation.json").read_text())
    boosted = json.loads((ROOT / "reports/terrain-fema-baselines.json").read_text())
    comparisons = {
        "fiveInputDnn": previous["testMetrics"],
        "initialTenInputDnn": initial_test_metrics,
        "tenInputBoosted": boosted["variants"]["full"]["testMetrics"],
    }
    report["comparisons"] = comparisons
    joblib.dump({
        "model": model, "imputer": imputer, "scaler": scaler,
        "features": FEATURES, "threshold": threshold,
    }, MODEL_PATH)
    ARTIFACT_PATH.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Ten-input terrain DNN evaluation", "",
        "The DNN uses the same ten inputs, FEMA map-class labels, and geographic splits "
        "as the terrain boosted model. Eighteen architecture and optimization candidates "
        "were ranked by geographic-validation ROC AUC; the decision threshold was "
        "selected on validation data. The selected network has 128, 64, and 32 hidden "
        "units, alpha 0.01, and no class weights. The app continues to serve the "
        "boosted model.", "",
        "| Geographic test metric | Five-input DNN | Initial terrain DNN | Tuned terrain DNN | Terrain boosted |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("Accuracy", "accuracy"), ("Balanced accuracy", "balancedAccuracy"),
        ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
        ("ROC AUC", "rocAuc"), ("Brier score", "brierScore"),
    ]:
        lines.append(f"| {label} | {comparisons['fiveInputDnn'][key]:.3f} | "
                     f"{comparisons['initialTenInputDnn'][key]:.3f} | "
                     f"{report['testMetrics'][key]:.3f} | "
                     f"{comparisons['tenInputBoosted'][key]:.3f} |")
    lines += ["", "These results measure FEMA map-class agreement. They do not "
              "measure the chance that a property floods during a specified storm, "
              "nor performance on addresses without a FEMA map class. The same "
              "test split has been used for earlier model comparisons, so these "
              "numbers are not independent confirmation on a fresh holdout.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Test balanced accuracy {report['testMetrics']['balancedAccuracy']:.4f}; "
          f"ROC AUC {report['testMetrics']['rocAuc']:.4f}")


if __name__ == "__main__":
    main()
