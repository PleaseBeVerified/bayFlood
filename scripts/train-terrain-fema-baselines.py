#!/usr/bin/env python3
"""Retrain FEMA map-class baselines with the available terrain features.

These artifacts are research comparisons, not storm-event flood predictions.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
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
from terrain_features import FEATURES, joined_rows


ROOT = Path(__file__).resolve().parents[1]
TRAINING_TABLE = ROOT / "data/dnn-training-table.csv"
CALIBRATION = ROOT / "data/physical-hazard-calibration.csv"
REPORT_JSON = ROOT / "reports/terrain-fema-baselines.json"
REPORT_MD = ROOT / "reports/terrain-fema-baselines.md"
APP_ARTIFACT = ROOT / "models/flood-risk-terrain-full.json"
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


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted)),
        "recall": float(recall_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted)),
        "rocAuc": float(roc_auc_score(labels, scores)),
        "brierScore": float(brier_score_loss(labels, scores)),
        "confusionMatrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
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
        "missing": "yes" if node["missing"] == node["yes"] else "no",
    }


def exported_predict(values: list[float], artifact: dict) -> float:
    values = np.asarray(values, dtype=np.float32)
    logit = artifact["baseLogit"]
    for tree in artifact["trees"]:
        node = tree
        while "leaf" not in node:
            value = values[node["feature"]]
            branch = node["missing"] if math.isnan(value) else (
                "yes" if value < node["threshold"] else "no"
            )
            node = node[branch]
        logit += node["leaf"]
    return 1 / (1 + math.exp(-logit))


def fit_variant(
    name: str,
    training_rows: list[dict[str, str]],
    x_by_id: dict[str, list[float]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> dict:
    def arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([x_by_id[row["location_id"]] for row in rows], dtype=np.float32),
            np.asarray([int(row["flood_risk_label"]) for row in rows], dtype=np.int64),
        )

    x_train, y_train = arrays(training_rows)
    x_validation, y_validation = arrays(validation_rows)
    x_test, y_test = arrays(test_rows)
    model = XGBClassifier(**PARAMETERS)
    started = perf_counter()
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - started
    validation_scores = model.predict_proba(x_validation)[:, 1]
    threshold = float(max(np.linspace(0.1, 0.9, 161), key=lambda candidate: (
        balanced_accuracy_score(y_validation, validation_scores >= candidate),
        f1_score(y_validation, validation_scores >= candidate),
    )))
    test_scores = model.predict_proba(x_test)[:, 1]
    model_path = ROOT / f"models/flood-risk-terrain-{name}.joblib"
    joblib.dump(model, model_path)
    gains = model.get_booster().get_score(importance_type="gain")
    result = {
        "modelPath": str(model_path.relative_to(ROOT)),
        "trainingRows": len(training_rows),
        "trainingSeconds": training_seconds,
        "trainingLabels": {
            "minimal": int((y_train == 0).sum()),
            "one_percent": int((y_train == 1).sum()),
        },
        "decisionThreshold": threshold,
        "validationMetrics": metrics(y_validation, validation_scores, threshold),
        "testMetrics": metrics(y_test, test_scores, threshold),
        "featureGain": {feature: float(gains.get(f"f{index}", 0))
                        for index, feature in enumerate(FEATURES)},
    }
    if name == "full":
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
        reference_inputs = [list(values) for values in x_test[:5]]
        reference_inputs.append([25, 60, 150, 1, math.nan, 2, 0, 0, math.log1p(500), 5])
        expected = model.predict_proba(np.asarray(reference_inputs, dtype=np.float32))[:, 1]
        parity_error = max(
            abs(exported_predict(values, artifact) - float(probability))
            for values, probability in zip(x_test, test_scores, strict=True)
        )
        parity_error = max(parity_error, *(
            abs(exported_predict(values, artifact) - float(probability))
            for values, probability in zip(reference_inputs, expected, strict=True)
        ))
        if parity_error > 0.00001:
            raise RuntimeError(f"Export parity error: {parity_error}")
        APP_ARTIFACT.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
        result["appArtifactPath"] = str(APP_ARTIFACT.relative_to(ROOT))
        result["exportMaximumProbabilityError"] = parity_error
        result["referenceCases"] = [
            {"inputs": [None if math.isnan(value) else float(value) for value in values],
             "probability": float(probability)}
            for values, probability in zip(reference_inputs, expected, strict=True)
        ]
    return result


def main() -> None:
    table_rows, x_by_id = joined_rows(TRAINING_TABLE, CALIBRATION)
    partitions = {
        name: [row for row in table_rows if row["spatial_partition"] == name]
        for name in ("training", "validation", "test")
    }
    if any(not rows for rows in partitions.values()):
        raise RuntimeError("Training, validation, and test must all be nonempty")
    training = partitions["training"]
    positives = [row for row in training if row["flood_risk_label"] == "1"]
    negatives = [row for row in training if row["flood_risk_label"] == "0"]
    high_negatives = sorted(
        negatives,
        key=lambda row: (-float(row["elevation_m"]), row["location_id"]),
    )[:len(positives)]
    selected_ids = {row["location_id"] for row in positives + high_negatives}
    high_training = [row for row in training if row["location_id"] in selected_ids]
    if len(high_training) != 2 * len(positives):
        raise RuntimeError("High-elevation training sample is not balanced")

    variants = {
        "full": fit_variant("full", training, x_by_id,
                            partitions["validation"], partitions["test"]),
        "high-elevation": fit_variant("high-elevation", high_training, x_by_id,
                                      partitions["validation"], partitions["test"]),
    }
    current = json.loads((ROOT / "reports/boosted-dnn-comparison.json").read_text())
    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "target": "FEMA mapped one_percent versus minimal; not observed storm inundation",
        "inputFiles": [str(TRAINING_TABLE.relative_to(ROOT)), str(CALIBRATION.relative_to(ROOT))],
        "features": FEATURES,
        "parameters": PARAMETERS,
        "partitionCounts": {name: len(rows) for name, rows in partitions.items()},
        "highElevationMinimalCutoffMeters": float(high_negatives[-1]["elevation_m"]),
        "currentAppTestMetrics": current["testMetrics"],
        "variants": variants,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Terrain-augmented FEMA map-class baselines", "",
        "These models add five terrain features to the current five-input FEMA classifier. "
        "They predict FEMA map classes, not flooding during a specified storm. "
        "Both variants use the same unchanged geographic validation and test splits. "
        "The high-elevation variant matches the number of positive training rows with "
        "the highest-elevation minimal rows; its cutoff is "
        f"{report['highElevationMinimalCutoffMeters']:.1f} m.", "",
        "| Geographic test metric | Previous app | Terrain, full training | Terrain, high-elevation sample |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balancedAccuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC AUC", "rocAuc"),
        ("Brier score", "brierScore"),
    ]:
        lines.append(
            f"| {label} | {current['testMetrics'][key]:.3f} | "
            f"{variants['full']['testMetrics'][key]:.3f} | "
            f"{variants['high-elevation']['testMetrics'][key]:.3f} |"
        )
    lines += ["", "The full-training terrain model is exported for the app. These test results cannot validate "
              "performance for locations without a FEMA map class or for observed storm events.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    for name, result in variants.items():
        print(f"{name}: validation balanced accuracy "
              f"{result['validationMetrics']['balancedAccuracy']:.3f}; "
              f"test balanced accuracy {result['testMetrics']['balancedAccuracy']:.3f}")


if __name__ == "__main__":
    main()
