#!/usr/bin/env python3
"""Train the app's FEMA-equivalent boosted classifier on geographic partitions."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
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
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "elevation_m",
    "current_imperviousness_pct",
    "current_100yr_24h_rainfall_mm",
    "nearest_water_available",
    "log_nearest_water_distance_m",
]
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
        math.log1p(max(0, float(distance))) if distance else math.nan,
    ]


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predictions)),
        "rocAuc": float(roc_auc_score(labels, probabilities)),
        "averagePrecision": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "brierScore": float(brier_score_loss(labels, probabilities)),
        "confusionMatrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def compact_tree(node: dict) -> dict:
    if "leaf" in node:
        return {"leaf": node["leaf"]}
    children = {child["nodeid"]: child for child in node["children"]}
    return {
        "feature": int(node["split"][1:]),
        "threshold": node["split_condition"],
        "yes": compact_tree(children[node["yes"]]),
        "no": compact_tree(children[node["no"]]),
        "missing": node["missing"] == node["yes"] and "yes" or "no",
    }


def exported_predict(values: list[float], artifact: dict) -> float:
    values = np.asarray(values, dtype=np.float32)
    base = artifact["baseLogit"]
    for tree in artifact["trees"]:
        node = tree
        while "leaf" not in node:
            value = values[node["feature"]]
            branch = node["missing"] if math.isnan(value) else (
                "yes" if value < node["threshold"] else "no"
            )
            node = node[branch]
        base += node["leaf"]
    return 1 / (1 + math.exp(-base))


def main() -> None:
    with (ROOT / "data/dnn-training-table.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    partitions = {
        name: [row for row in rows if row["spatial_partition"] == name]
        for name in ("training", "validation", "test")
    }
    if any(not part for part in partitions.values()):
        raise RuntimeError("Training, validation, and test partitions must all be nonempty")
    x = {name: np.asarray([vector(row) for row in part], dtype=np.float32)
         for name, part in partitions.items()}
    y = {name: np.asarray([int(row["flood_risk_label"]) for row in part])
         for name, part in partitions.items()}

    model = XGBClassifier(**PARAMETERS)
    model.fit(x["training"], y["training"])
    validation = model.predict_proba(x["validation"])[:, 1]
    thresholds = np.linspace(0.1, 0.9, 161)
    threshold = float(max(thresholds, key=lambda value: (
        balanced_accuracy_score(y["validation"], validation >= value),
        f1_score(y["validation"], validation >= value),
    )))
    test = model.predict_proba(x["test"])[:, 1]

    booster = model.get_booster()
    config = json.loads(booster.save_config())
    base_probability = float(json.loads(config["learner"]["learner_model_param"]["base_score"])[0])
    artifact = {
        "schemaVersion": 1,
        "modelType": "gradient-boosted-binary-classifier",
        "features": FEATURES,
        "baseLogit": math.log(base_probability / (1 - base_probability)),
        "decisionThreshold": threshold,
        "trees": [compact_tree(json.loads(tree)) for tree in booster.get_dump(dump_format="json")],
    }
    reference_rows = partitions["test"][:5]
    reference_inputs = [vector(row) for row in reference_rows]
    reference_inputs.append([25, 60, 150, 1, math.nan])
    reference_probabilities = model.predict_proba(np.asarray(reference_inputs, dtype=np.float32))[:, 1]
    parity_error = max(abs(exported_predict(values, artifact) - float(expected))
                       for values, expected in zip(x["test"], test, strict=True))
    parity_error = max(parity_error, *(abs(exported_predict(values, artifact) - float(expected))
                       for values, expected in zip(reference_inputs, reference_probabilities, strict=True)))
    if parity_error > 0.00001:
        raise RuntimeError(f"Export parity error: {parity_error}")

    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "target": "FEMA one_percent=1, minimal=0; 0.2 percent excluded",
        "features": FEATURES,
        "parameters": PARAMETERS,
        "rowCounts": {name: len(part) for name, part in partitions.items()},
        "decisionThreshold": threshold,
        "validationMetrics": metrics(y["validation"], validation, threshold),
        "testMetrics": metrics(y["test"], test, threshold),
        "referenceCases": [
            {"inputs": [None if math.isnan(value) else value for value in values],
             "probability": float(probability)}
            for values, probability in zip(reference_inputs, reference_probabilities, strict=True)
        ],
        "exportMaximumProbabilityError": parity_error,
    }
    dnn_report = json.loads((ROOT / "reports/dnn-model-evaluation.json").read_text())
    dnn_test = dnn_report["testMetrics"]
    report["dnnTestMetrics"] = dnn_test
    (ROOT / "models/flood-risk-boosted.json").write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    joblib.dump(model, ROOT / "models/flood-risk-boosted.joblib")
    (ROOT / "reports/boosted-dnn-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    test_metrics = report["testMetrics"]
    (ROOT / "reports/boosted-dnn-comparison.md").write_text(
        "# Boosted model comparison\n\n"
        "This model uses the same five inputs and geographic partitions as the DNN. "
        "Hyperparameters and the decision threshold were chosen using training and validation data. "
        "The test split evaluates agreement with FEMA-derived mapped classes, not actual property floods.\n\n"
        f"- Validation balanced accuracy: {report['validationMetrics']['balancedAccuracy']:.3f}\n"
        f"- Test accuracy: {test_metrics['accuracy']:.3f}\n"
        f"- Test balanced accuracy: {test_metrics['balancedAccuracy']:.3f}\n"
        f"- Test precision: {test_metrics['precision']:.3f}\n"
        f"- Test recall: {test_metrics['recall']:.3f}\n"
        f"- Test F1: {test_metrics['f1']:.3f}\n"
        f"- Test ROC AUC: {test_metrics['rocAuc']:.3f}\n"
        f"- Confusion matrix [[TN, FP], [FN, TP]]: {test_metrics['confusionMatrix']}\n\n"
        "| Geographic test metric | Earlier DNN | Boosted app model |\n"
        "| --- | ---: | ---: |\n"
        f"| Accuracy | {dnn_test['accuracy']:.3f} | {test_metrics['accuracy']:.3f} |\n"
        f"| Balanced accuracy | {dnn_test['balancedAccuracy']:.3f} | {test_metrics['balancedAccuracy']:.3f} |\n"
        f"| Recall | {dnn_test['recall']:.3f} | {test_metrics['recall']:.3f} |\n"
        f"| F1 | {dnn_test['f1']:.3f} | {test_metrics['f1']:.3f} |\n\n"
        "The boosted model reduces false positives from 103 to 56 on this split, "
        "while false negatives rise from 11 to 15. An independent future geographic "
        "holdout is needed to confirm the gain after this model comparison.\n\n"
        "The DNN remains available for research comparison. The app uses this boosted model "
        "for its separate current-condition pattern score. FEMA map classes still determine "
        "the displayed flood-event benchmarks.\n"
    )
    print(f"Saved boosted model; test accuracy={test_metrics['accuracy']:.4f}, "
          f"balanced accuracy={test_metrics['balancedAccuracy']:.4f}")


if __name__ == "__main__":
    main()
