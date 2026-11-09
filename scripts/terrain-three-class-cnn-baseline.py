#!/usr/bin/env python3
"""Train the three-class boosted comparator on the CNN's buffered rows."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from terrain_features import joined_rows
from terrain_three_class_metrics import CLASSES, evaluate


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/terrain-cnn/three-class"


def main() -> None:
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    rows = metadata["rows"]
    table, vectors = joined_rows(
        ROOT / "data/dnn-three-class-training-table.csv",
        ROOT / "data/physical-hazard-calibration.csv", "mapped_hazard_class",
    )
    if [r["location_id"] for r in rows] != [r["location_id"] for r in table]:
        raise RuntimeError("CNN patch cache order differs from the three-class table")
    partition = json.loads((CACHE / "partitions.json").read_text(encoding="utf-8"))
    indices = {name: np.asarray(partition[name], dtype=np.int64)
               for name in ("training", "validation", "test")}
    x = np.asarray([vectors[r["location_id"]] for r in rows], dtype=np.float32)
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    if any(set(y[part]) != {0, 1, 2} for part in indices.values()):
        raise RuntimeError("A buffered partition is missing a class")

    candidates = []
    models = []
    for balanced in (False, True):
        model = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.9, reg_lambda=10,
            n_jobs=4, random_state=42, objective="multi:softprob", num_class=3,
        )
        weights = (compute_sample_weight(class_weight="balanced", y=y[indices["training"]])
                   if balanced else None)
        model.fit(x[indices["training"]], y[indices["training"]], sample_weight=weights)
        validation = evaluate(y[indices["validation"]],
                              model.predict_proba(x[indices["validation"]]))
        candidates.append({"classWeight": "balanced" if balanced else "none",
                           "validation": validation})
        models.append(model)
        print(f"Boosted balanced={balanced}: validation macro F1 "
              f"{validation['macroF1']:.3f}", flush=True)

    selected = max(range(len(models)), key=lambda i: (
        candidates[i]["validation"]["macroF1"],
        candidates[i]["validation"]["balancedAccuracy"],
    ))
    test = evaluate(y[indices["test"]],
                    models[selected].predict_proba(x[indices["test"]]))
    joblib.dump(models[selected], CACHE / "baseline.joblib")
    (CACHE / "baseline.json").write_text(json.dumps({
        "classes": CLASSES,
        "hyperparameters": {"trees": 500, "maxDepth": 4, "learningRate": 0.03,
                            "subsample": 0.85, "colsampleBytree": 0.9, "regLambda": 10},
        "candidates": candidates,
        "selected": candidates[selected],
        "test": test,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Buffered boosted test macro F1 {test['macroF1']:.3f}; "
          f"0.2% F1 {test['perClass']['point_two_percent']['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
