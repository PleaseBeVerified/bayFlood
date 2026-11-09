#!/usr/bin/env python3
"""Compare terrain boosted classifiers using all three FEMA mapped classes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from terrain_features import FEATURES, joined_rows


ROOT = Path(__file__).resolve().parents[1]
TRAINING_TABLE = ROOT / "data/dnn-three-class-training-table.csv"
CALIBRATION = ROOT / "data/physical-hazard-calibration.csv"
MODEL_PATH = ROOT / "models/flood-hazard-terrain-three-class-boosted.joblib"
REPORT_JSON = ROOT / "reports/terrain-three-class-boosted-evaluation.json"
REPORT_MD = ROOT / "reports/terrain-three-class-boosted-evaluation.md"
CLASSES = ["minimal", "one_percent", "point_two_percent"]
CANDIDATES = [
    (350, 3, False), (350, 3, True),
    (500, 3, True), (500, 4, True),
    (350, 4, False), (500, 4, False),
    (500, 5, True), (500, 5, False),
]


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=[0, 1, 2], zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balancedAccuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macroF1": float(f1_score(y_true, predicted, average="macro")),
        "logLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "perClass": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASSES)
        },
        "confusionMatrix": confusion_matrix(y_true, predicted, labels=[0, 1, 2]).tolist(),
    }


def main() -> None:
    rows, x_by_id = joined_rows(TRAINING_TABLE, CALIBRATION, "mapped_hazard_class")
    partitions = {
        name: [row for row in rows if row["spatial_partition"] == name]
        for name in ("training", "validation", "test")
    }
    if any({row["mapped_hazard_class"] for row in values} != set(CLASSES)
           for values in partitions.values()):
        raise RuntimeError("Every geographic partition must contain all three classes")
    x = {
        name: np.asarray([x_by_id[row["location_id"]] for row in values], dtype=np.float32)
        for name, values in partitions.items()
    }
    y = {
        name: np.asarray([CLASSES.index(row["mapped_hazard_class"]) for row in values])
        for name, values in partitions.items()
    }
    balanced_weights = compute_sample_weight(class_weight="balanced", y=y["training"])

    candidates = []
    models = []
    for tree_count, depth, balanced in CANDIDATES:
        model = XGBClassifier(
            n_estimators=tree_count,
            max_depth=depth,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=10,
            n_jobs=4,
            random_state=42,
            objective="multi:softprob",
            num_class=3,
        )
        model.fit(
            x["training"], y["training"],
            sample_weight=balanced_weights if balanced else None,
        )
        validation = metrics(y["validation"], model.predict_proba(x["validation"]))
        candidates.append({
            "trees": tree_count,
            "maxDepth": depth,
            "classWeight": "balanced" if balanced else "none",
            "validationMacroF1": validation["macroF1"],
            "validationBalancedAccuracy": validation["balancedAccuracy"],
            "validationPointTwoF1": validation["perClass"]["point_two_percent"]["f1"],
        })
        models.append(model)
        print(f"Candidate {tree_count} trees, depth {depth}, balanced {balanced}: "
              f"validation macro F1 {validation['macroF1']:.3f}", flush=True)

    selected = max(range(len(models)), key=lambda index: (
        candidates[index]["validationMacroF1"],
        candidates[index]["validationBalancedAccuracy"],
    ))
    model = models[selected]
    validation_metrics = metrics(y["validation"], model.predict_proba(x["validation"]))
    test_metrics = metrics(y["test"], model.predict_proba(x["test"]))
    old = json.loads((ROOT / "reports/three-class-dnn-evaluation.json").read_text())
    terrain_dnn = json.loads((ROOT / "reports/terrain-three-class-dnn-evaluation.json").read_text())
    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "target": "FEMA mapped class; not observed flooding",
        "inputFiles": [str(TRAINING_TABLE.relative_to(ROOT)), str(CALIBRATION.relative_to(ROOT))],
        "features": FEATURES,
        "classes": CLASSES,
        "partitionCounts": {name: len(values) for name, values in partitions.items()},
        "labelCounts": {
            name: {label: int((y[name] == index).sum()) for index, label in enumerate(CLASSES)}
            for name in partitions
        },
        "candidates": candidates,
        "selectedCandidate": candidates[selected],
        "selectionCriterion": "highest geographic-validation macro F1, then balanced accuracy",
        "validationMetrics": validation_metrics,
        "testMetrics": test_metrics,
        "fiveInputDnnTestMetrics": old["testMetrics"],
        "terrainDnnTestMetrics": terrain_dnn["testMetrics"],
    }
    joblib.dump(model, MODEL_PATH)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Three-class terrain boosted evaluation", "",
        "The model uses all 10,000 FEMA-labeled samples, including 933 mapped "
        "0.2% annual-chance locations, with the ten terrain inputs. Eight "
        "candidates were ranked by geographic-validation macro F1. The app "
        "continues to serve the binary terrain boosted score.", "",
        "| Geographic test metric | Five-input DNN | Terrain DNN | Terrain boosted |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("Accuracy", "accuracy"), ("Balanced accuracy", "balancedAccuracy"),
        ("Macro F1", "macroF1"), ("Log loss", "logLoss"),
    ]:
        lines.append(f"| {label} | {old['testMetrics'][key]:.3f} | "
                     f"{terrain_dnn['testMetrics'][key]:.3f} | {test_metrics[key]:.3f} |")
    lines += ["", "| Terrain boosted test class | Samples | Precision | Recall | F1 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in CLASSES:
        result = test_metrics["perClass"][name]
        lines.append(f"| {name} | {result['support']} | {result['precision']:.3f} | "
                     f"{result['recall']:.3f} | {result['f1']:.3f} |")
    lines += ["", "Confusion matrix (rows observed, columns predicted; class order "
              f"{', '.join(CLASSES)}): `{test_metrics['confusionMatrix']}`.", "",
              "The 0.2% class remains difficult to distinguish. These are FEMA "
              "map classes, not observed flood events; an unmapped address has no "
              "independent label in this evaluation.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Test macro F1 {test_metrics['macroF1']:.3f}; "
          f"0.2% F1 {test_metrics['perClass']['point_two_percent']['f1']:.3f}")


if __name__ == "__main__":
    main()
