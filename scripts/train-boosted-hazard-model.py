#!/usr/bin/env python3
"""Train an honest XGBoost benchmark for the physical flood-hazard model.

It uses a deterministic geographic split, selects hyperparameters on validation
blocks, reports results once on untouched test blocks, and exports dependency-
free trees for the TypeScript production evaluator.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "physical-hazard-calibration.csv"
LOGISTIC_MODEL_PATH = ROOT / "models" / "flood-physical-hazard-model.json"
PRODUCTION_MODEL_PATH = ROOT / "models" / "flood-physical-hazard-model.json"
ONE_PERCENT_MODEL_PATH = (
    ROOT / "models" / "flood-physical-hazard-boosted-one-percent.json"
)
MODERATE_MODEL_PATH = (
    ROOT / "models" / "flood-physical-hazard-boosted-moderate.json"
)
METADATA_PATH = ROOT / "models" / "flood-physical-hazard-boosted-metadata.json"
REPORT_JSON_PATH = ROOT / "reports" / "boosted-model-evaluation.json"
REPORT_MARKDOWN_PATH = ROOT / "reports" / "boosted-model-evaluation.md"
PARITY_FIXTURE_PATH = ROOT / "tests" / "fixtures" / "physical-boosted-parity.json"

SPATIAL_BLOCK_DEGREES = 0.05
CLASS_ORDER = ["one_percent", "point_two_percent", "minimal"]
NUMERIC_FEATURES = [
    "rainfall_100yr_24h_in",
    "log_elevation_m",
    "log_nearest_water_distance_m",
    "slope_degrees",
    "local_elevation_250m",
    "local_elevation_500m",
    "log_flow_accumulation_m2",
    "log_height_above_drainage_m",
]
CATEGORICAL_FEATURE = "nearest_water_kind"
MONOTONIC_DIRECTIONS = {
    "rainfall_100yr_24h_in": 1,
    "log_elevation_m": -1,
    "log_nearest_water_distance_m": -1,
    "slope_degrees": 0,
    "local_elevation_250m": -1,
    "local_elevation_500m": -1,
    "log_flow_accumulation_m2": 1,
    "log_height_above_drainage_m": -1,
}
TARGETS = {
    "onePercent": "one_percent",
    "atLeastModerate": "one_percent_or_point_two_percent",
}
HYPERPARAMETER_GRID = [
    {
        "max_depth": 2,
        "eta": 0.03,
        "min_child_weight": 5,
        "reg_lambda": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "max_depth": 3,
        "eta": 0.03,
        "min_child_weight": 10,
        "reg_lambda": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "max_depth": 3,
        "eta": 0.05,
        "min_child_weight": 20,
        "reg_lambda": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "max_depth": 4,
        "eta": 0.03,
        "min_child_weight": 20,
        "reg_lambda": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
]
MAXIMUM_ROUNDS = 2_000
EARLY_STOPPING_ROUNDS = 75
SEED = 42
ANNUAL_CHANCE_ANCHORS = {
    "one_percent": 0.01,
    "point_two_percent": 0.002,
    "minimal": 0.001,
}


def parse_optional_number(value: str | None) -> float:
    if value is None or value.strip() == "":
        return math.nan
    parsed = float(value)
    return parsed if math.isfinite(parsed) else math.nan


def numeric_value(row: dict[str, str], feature: str) -> float:
    raw_feature = {
        "log_elevation_m": "elevation_m",
        "log_nearest_water_distance_m": "nearest_water_distance_m",
        "log_flow_accumulation_m2": "flow_accumulation_m2",
        "log_height_above_drainage_m": "height_above_drainage_m",
    }.get(feature, feature)
    value = parse_optional_number(row.get(raw_feature))
    if math.isnan(value):
        return value
    if feature.startswith("log_"):
        return math.log1p(max(0.0, value))
    return value


def fnv1a_32(value: str) -> int:
    hash_value = 2_166_136_261
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 16_777_619) & 0xFFFFFFFF
    return hash_value


def spatial_partition(row: dict[str, str]) -> str:
    latitude_tile = math.floor(float(row["lat"]) / SPATIAL_BLOCK_DEGREES)
    longitude_tile = math.floor(float(row["lng"]) / SPATIAL_BLOCK_DEGREES)
    bucket = fnv1a_32(f"{latitude_tile},{longitude_tile}") % 20
    if bucket < 14:
        return "training"
    if bucket < 17:
        return "validation"
    return "test"


def load_rows() -> list[dict[str, str]]:
    with INPUT_PATH.open(newline="", encoding="utf-8") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["mapped_hazard_class"] in CLASS_ORDER
        ]
    if len(rows) < 1_000:
        raise RuntimeError(f"{INPUT_PATH} needs at least 1,000 valid rows")
    return rows


def categories_for(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row.get(CATEGORICAL_FEATURE) or "Unknown" for row in rows})


def feature_names(categories: list[str]) -> list[str]:
    return [
        *NUMERIC_FEATURES,
        *(f"nearest_water_kind_{index}" for index, _ in enumerate(categories)),
    ]


def vectorize(
    rows: list[dict[str, str]], categories: list[str]
) -> np.ndarray[Any, np.dtype[np.float32]]:
    category_indexes = {category: index for index, category in enumerate(categories)}
    matrix = np.zeros((len(rows), len(NUMERIC_FEATURES) + len(categories)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for feature_index, feature in enumerate(NUMERIC_FEATURES):
            matrix[row_index, feature_index] = numeric_value(row, feature)
        category = row.get(CATEGORICAL_FEATURE) or "Unknown"
        category_index = category_indexes.get(category)
        if category_index is not None:
            matrix[row_index, len(NUMERIC_FEATURES) + category_index] = 1.0
    return matrix


def binary_labels(rows: list[dict[str, str]], target: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    if target == "onePercent":
        values = [row["mapped_hazard_class"] == "one_percent" for row in rows]
    else:
        values = [row["mapped_hazard_class"] != "minimal" for row in rows]
    return np.asarray(values, dtype=np.float32)


def multiclass_labels(rows: list[dict[str, str]]) -> list[str]:
    return [row["mapped_hazard_class"] for row in rows]


def base_parameters(
    candidate: dict[str, float | int], feature_count: int
) -> dict[str, Any]:
    directions = [
        *(MONOTONIC_DIRECTIONS[feature] for feature in NUMERIC_FEATURES),
        *(0 for _ in range(feature_count - len(NUMERIC_FEATURES))),
    ]
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "max_bin": 512,
        "seed": SEED,
        "nthread": 0,
        "monotone_constraints": tuple(directions),
        **candidate,
    }


def binary_metrics(actual: np.ndarray[Any, Any], probability: np.ndarray[Any, Any]) -> dict[str, Any]:
    return {
        "rows": int(actual.size),
        "positives": int(actual.sum()),
        "rocAuc": float(roc_auc_score(actual, probability)),
        "brierScore": float(brier_score_loss(actual, probability)),
    }


def tune_target(
    target: str,
    train_matrix: np.ndarray[Any, Any],
    train_rows: list[dict[str, str]],
    validation_matrix: np.ndarray[Any, Any],
    validation_rows: list[dict[str, str]],
    names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_labels = binary_labels(train_rows, target)
    validation_labels = binary_labels(validation_rows, target)
    train_data = xgb.DMatrix(train_matrix, label=train_labels, feature_names=names)
    validation_data = xgb.DMatrix(
        validation_matrix, label=validation_labels, feature_names=names
    )
    leaderboard: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(HYPERPARAMETER_GRID, start=1):
        model = xgb.train(
            base_parameters(candidate, len(names)),
            train_data,
            num_boost_round=MAXIMUM_ROUNDS,
            evals=[(validation_data, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        rounds = model.best_iteration + 1
        probability = model.predict(validation_data, iteration_range=(0, rounds))
        metrics = binary_metrics(validation_labels, probability)
        entry = {
            "candidate": candidate_index,
            "hyperparameters": candidate,
            "rounds": rounds,
            "validation": metrics,
        }
        leaderboard.append(entry)
        print(
            f"  candidate {candidate_index}/{len(HYPERPARAMETER_GRID)}: "
            f"rounds={rounds}, AUC={metrics['rocAuc']:.4f}, "
            f"Brier={metrics['brierScore']:.4f}",
            flush=True,
        )
    # The app outputs probabilities, so select by the proper probability-scoring
    # rule first. AUC only breaks ties between similarly calibrated candidates.
    leaderboard.sort(
        key=lambda entry: (
            entry["validation"]["brierScore"],
            -entry["validation"]["rocAuc"],
        )
    )
    return leaderboard[0], leaderboard


def train_final_target(
    target: str,
    selected: dict[str, Any],
    model_matrix: np.ndarray[Any, Any],
    model_rows: list[dict[str, str]],
    names: list[str],
) -> xgb.Booster:
    model_data = xgb.DMatrix(
        model_matrix, label=binary_labels(model_rows, target), feature_names=names
    )
    return xgb.train(
        base_parameters(selected["hyperparameters"], len(names)),
        model_data,
        num_boost_round=selected["rounds"],
        verbose_eval=False,
    )


def ordered_probabilities(
    one_percent_probability: np.ndarray[Any, Any],
    moderate_probability: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    at_least_moderate = np.maximum(one_percent_probability, moderate_probability)
    return np.column_stack(
        (
            one_percent_probability,
            at_least_moderate - one_percent_probability,
            1.0 - at_least_moderate,
        )
    )


def multiclass_metrics(
    actual: list[str], class_probabilities: np.ndarray[Any, Any]
) -> dict[str, Any]:
    predicted = [CLASS_ORDER[index] for index in np.argmax(class_probabilities, axis=1)]
    report = classification_report(
        actual,
        predicted,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balancedAccuracy": float(balanced_accuracy_score(actual, predicted)),
        "macroF1": float(f1_score(actual, predicted, labels=CLASS_ORDER, average="macro")),
        "confusionMatrix": confusion_matrix(actual, predicted, labels=CLASS_ORDER).tolist(),
        "classOrder": CLASS_ORDER,
        "perClass": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in CLASS_ORDER
        },
    }


def evaluate_boosted(
    rows: list[dict[str, str]],
    matrix: np.ndarray[Any, Any],
    names: list[str],
    one_percent_model: xgb.Booster,
    moderate_model: xgb.Booster,
) -> dict[str, Any]:
    data = xgb.DMatrix(matrix, feature_names=names)
    one_probability = one_percent_model.predict(data)
    moderate_probability = moderate_model.predict(data)
    return {
        "onePercent": binary_metrics(binary_labels(rows, "onePercent"), one_probability),
        "atLeastModerate": binary_metrics(
            binary_labels(rows, "atLeastModerate"), moderate_probability
        ),
        "classification": multiclass_metrics(
            multiclass_labels(rows), ordered_probabilities(one_probability, moderate_probability)
        ),
    }


def logistic_numeric_value(row: dict[str, str], feature: str) -> float | None:
    value = numeric_value(row, feature)
    return None if math.isnan(value) else value


def sigmoid(value: float) -> float:
    if value < -35:
        return 0.0
    if value > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def evaluate_logistic_artifact(rows: list[dict[str, str]]) -> dict[str, Any]:
    with LOGISTIC_MODEL_PATH.open(encoding="utf-8") as source:
        artifact = json.load(source)
    if artifact.get("modelType") != "ordered_logistic_physical_hazard":
        raise ValueError("The production artifact is no longer the logistic model")
    preprocessor = artifact["preprocessor"]
    numeric_features = artifact["numericFeatures"]

    def predict(row: dict[str, str], weights: list[float]) -> float:
        vector: list[float] = []
        for feature in numeric_features:
            stats = preprocessor["numericStats"][feature]
            value = logistic_numeric_value(row, feature)
            if value is None:
                value = stats["median"]
            vector.append((value - stats["mean"]) / stats["standardDeviation"])
        category = row.get(CATEGORICAL_FEATURE) or "Unknown"
        vector.extend(
            float(category == known)
            for known in preprocessor["categories"][CATEGORICAL_FEATURE]
        )
        return sigmoid(weights[0] + sum(value * weight for value, weight in zip(vector, weights[1:])))

    one_probability = np.asarray(
        [predict(row, artifact["models"]["onePercent"]["weights"]) for row in rows]
    )
    moderate_probability = np.asarray(
        [predict(row, artifact["models"]["atLeastModerate"]["weights"]) for row in rows]
    )
    return {
        "onePercent": binary_metrics(binary_labels(rows, "onePercent"), one_probability),
        "atLeastModerate": binary_metrics(
            binary_labels(rows, "atLeastModerate"), moderate_probability
        ),
        "classification": multiclass_metrics(
            multiclass_labels(rows), ordered_probabilities(one_probability, moderate_probability)
        ),
    }


def previous_logistic_baseline(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Keep an evaluation-only baseline after the old model weights are removed."""
    try:
        return evaluate_logistic_artifact(rows)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        if REPORT_JSON_PATH.exists():
            with REPORT_JSON_PATH.open(encoding="utf-8") as source:
                previous_report = json.load(source)
            baseline = previous_report.get(
                "previousLogisticBaseline", previous_report.get("currentLogistic")
            )
            if isinstance(baseline, dict) and isinstance(baseline.get("test"), dict):
                return baseline["test"]
        raise RuntimeError(
            "The old logistic model is gone and no historical baseline exists in "
            f"{REPORT_JSON_PATH.relative_to(ROOT)}"
        )


def compact_booster(model: xgb.Booster) -> dict[str, Any]:
    native = json.loads(model.save_raw(raw_format="json"))
    learner = native["learner"]
    raw_base_score = learner["learner_model_param"]["base_score"]
    base_score = float(raw_base_score.strip("[]"))
    trees = learner["gradient_booster"]["model"]["trees"]
    return {
        "baseScore": base_score,
        "trees": [
            {
                "leftChildren": tree["left_children"],
                "rightChildren": tree["right_children"],
                "splitIndices": tree["split_indices"],
                "splitConditions": tree["split_conditions"],
                "defaultLeft": tree["default_left"],
            }
            for tree in trees
        ],
    }


def numeric_ranges(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for feature in NUMERIC_FEATURES:
        values = [numeric_value(row, feature) for row in rows]
        finite_values = [value for value in values if math.isfinite(value)]
        ranges[feature] = {
            "minimum": min(finite_values),
            "maximum": max(finite_values),
        }
    return ranges


def input_from_row(row: dict[str, str]) -> dict[str, Any]:
    def value(column: str) -> float | None:
        parsed = parse_optional_number(row.get(column))
        return None if math.isnan(parsed) else parsed

    return {
        "elevation_m": value("elevation_m"),
        "rainfall_100yr_24h_in": value("rainfall_100yr_24h_in"),
        "nearest_water_distance_m": value("nearest_water_distance_m"),
        "nearest_water_kind": row.get("nearest_water_kind") or "Unknown",
        "slope_degrees": value("slope_degrees"),
        "local_elevation_250m": value("local_elevation_250m"),
        "local_elevation_500m": value("local_elevation_500m"),
        "flow_accumulation_m2": value("flow_accumulation_m2"),
        "height_above_drainage_m": value("height_above_drainage_m"),
    }


def write_parity_fixture(
    rows: list[dict[str, str]],
    matrix: np.ndarray[Any, Any],
    names: list[str],
    one_percent_model: xgb.Booster,
    moderate_model: xgb.Booster,
) -> None:
    selected_indexes = list(range(min(12, len(rows))))
    missing_distance_index = next(
        (
            index
            for index, row in enumerate(rows)
            if not row.get("nearest_water_distance_m", "").strip()
        ),
        None,
    )
    if missing_distance_index is not None and missing_distance_index not in selected_indexes:
        selected_indexes.append(missing_distance_index)
    selected_matrix = matrix[selected_indexes]
    data = xgb.DMatrix(selected_matrix, feature_names=names)
    one_probabilities = one_percent_model.predict(data)
    moderate_probabilities = moderate_model.predict(data)
    fixtures = []
    for fixture_index, row_index in enumerate(selected_indexes):
        one_probability = float(one_probabilities[fixture_index])
        at_least_moderate = max(
            one_probability, float(moderate_probabilities[fixture_index])
        )
        fixtures.append(
            {
                "input": input_from_row(rows[row_index]),
                "expectedClassProbabilities": {
                    "onePercent": one_probability,
                    "pointTwoPercent": at_least_moderate - one_probability,
                    "minimal": 1.0 - at_least_moderate,
                },
            }
        )
    PARITY_FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARITY_FIXTURE_PATH.write_text(
        json.dumps({"tolerance": 2e-6, "fixtures": fixtures}, indent=2) + "\n",
        encoding="utf-8",
    )


def markdown_report(report: dict[str, Any]) -> str:
    boosted = report["boosted"]["test"]
    logistic = report["previousLogisticBaseline"]["test"]
    winner = report["comparison"]["recommendation"]
    lines = [
        "# Gradient-boosted physical hazard benchmark",
        "",
        "The untouched test set contains geographic blocks that were not used for training or hyperparameter selection.",
        "Hyperparameters were selected by the lowest validation Brier score because the app outputs probabilities; AUC breaks ties.",
        "",
        "| Metric | Current logistic | Gradient boosted | Change |",
        "| --- | ---: | ---: | ---: |",
        f"| Class accuracy | {logistic['classification']['accuracy']:.3%} | {boosted['classification']['accuracy']:.3%} | {boosted['classification']['accuracy'] - logistic['classification']['accuracy']:+.3%} |",
        f"| Balanced accuracy | {logistic['classification']['balancedAccuracy']:.3%} | {boosted['classification']['balancedAccuracy']:.3%} | {boosted['classification']['balancedAccuracy'] - logistic['classification']['balancedAccuracy']:+.3%} |",
        f"| Macro F1 | {logistic['classification']['macroF1']:.4f} | {boosted['classification']['macroF1']:.4f} | {boosted['classification']['macroF1'] - logistic['classification']['macroF1']:+.4f} |",
        f"| 1% AUC | {logistic['onePercent']['rocAuc']:.4f} | {boosted['onePercent']['rocAuc']:.4f} | {boosted['onePercent']['rocAuc'] - logistic['onePercent']['rocAuc']:+.4f} |",
        f"| >=0.2% AUC | {logistic['atLeastModerate']['rocAuc']:.4f} | {boosted['atLeastModerate']['rocAuc']:.4f} | {boosted['atLeastModerate']['rocAuc'] - logistic['atLeastModerate']['rocAuc']:+.4f} |",
        f"| 1% Brier (lower is better) | {logistic['onePercent']['brierScore']:.4f} | {boosted['onePercent']['brierScore']:.4f} | {boosted['onePercent']['brierScore'] - logistic['onePercent']['brierScore']:+.4f} |",
        f"| >=0.2% Brier (lower is better) | {logistic['atLeastModerate']['brierScore']:.4f} | {boosted['atLeastModerate']['brierScore']:.4f} | {boosted['atLeastModerate']['brierScore'] - logistic['atLeastModerate']['brierScore']:+.4f} |",
        "",
        "## Class recall",
        "",
        "| FEMA-equivalent class | Current logistic | Gradient boosted |",
        "| --- | ---: | ---: |",
        *(f"| {label} | {logistic['classification']['perClass'][label]['recall']:.3%} | {boosted['classification']['perClass'][label]['recall']:.3%} |" for label in CLASS_ORDER),
        "",
        f"Recommendation: **{winner}**",
        "",
        "Important limitation: the middle 0.2% class remains difficult to identify. These labels are FEMA-equivalent hazard classes, not observed 100-year flood outcomes at individual properties.",
        "",
        "The gradient-boosted trees are exported to the production model artifact used by the app.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    splits = {
        name: [row for row in rows if spatial_partition(row) == name]
        for name in ("training", "validation", "test")
    }
    if any(len(split) < 100 for split in splits.values()):
        raise RuntimeError("A geographic split has too few rows for evaluation")
    print(
        "Geographic split:",
        {name: len(split) for name, split in splits.items()},
        flush=True,
    )

    categories = categories_for(splits["training"])
    names = feature_names(categories)
    matrices = {name: vectorize(split, categories) for name, split in splits.items()}

    selected: dict[str, dict[str, Any]] = {}
    leaderboards: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        print(f"Tuning {target}...", flush=True)
        selected[target], leaderboards[target] = tune_target(
            target,
            matrices["training"],
            splits["training"],
            matrices["validation"],
            splits["validation"],
            names,
        )

    model_rows = [*splits["training"], *splits["validation"]]
    model_matrix = vectorize(model_rows, categories)
    print("Training final models on training + validation rows...", flush=True)
    one_percent_model = train_final_target(
        "onePercent", selected["onePercent"], model_matrix, model_rows, names
    )
    moderate_model = train_final_target(
        "atLeastModerate",
        selected["atLeastModerate"],
        model_matrix,
        model_rows,
        names,
    )

    boosted_fitted = evaluate_boosted(
        model_rows, model_matrix, names, one_percent_model, moderate_model
    )
    boosted_test = evaluate_boosted(
        splits["test"],
        matrices["test"],
        names,
        one_percent_model,
        moderate_model,
    )
    logistic_test = previous_logistic_baseline(splits["test"])

    accuracy_change = (
        boosted_test["classification"]["accuracy"]
        - logistic_test["classification"]["accuracy"]
    )
    auc_changes = {
        target: boosted_test[target]["rocAuc"] - logistic_test[target]["rocAuc"]
        for target in ("onePercent", "atLeastModerate")
    }
    brier_changes = {
        target: boosted_test[target]["brierScore"]
        - logistic_test[target]["brierScore"]
        for target in ("onePercent", "atLeastModerate")
    }
    should_integrate = (
        accuracy_change > 0
        and min(auc_changes.values()) >= 0
        and max(brier_changes.values()) <= 0
    )
    recommendation = (
        "use the gradient-boosted production model with the parity test retained"
        if should_integrate
        else "keep the current production model; the boosted benchmark did not improve every required test metric"
    )

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceTable": str(INPUT_PATH.relative_to(ROOT)),
        "split": {
            "method": "deterministic 0.05-degree geographic blocks",
            **{f"{name}Rows": len(split) for name, split in splits.items()},
        },
        "classCounts": dict(Counter(row["mapped_hazard_class"] for row in rows)),
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": {CATEGORICAL_FEATURE: categories},
            "expandedNames": names,
            "monotonicDirections": [
                *(MONOTONIC_DIRECTIONS[name] for name in NUMERIC_FEATURES),
                *(0 for _ in categories),
            ],
        },
        "selection": {"selected": selected, "leaderboards": leaderboards},
        "boosted": {"fitted": boosted_fitted, "test": boosted_test},
        "previousLogisticBaseline": {"test": logistic_test},
        "comparison": {
            "testAccuracyChange": accuracy_change,
            "testAucChanges": auc_changes,
            "testBrierChanges": brier_changes,
            "passesAutomaticIntegrationGate": should_integrate,
            "recommendation": recommendation,
        },
        "importantLimit": (
            "These are FEMA-equivalent physical hazard classifications, not observed "
            "100-year flood outcomes at individual properties."
        ),
    }
    metadata = {
        "schemaVersion": 1,
        "modelType": "ordered_xgboost_physical_hazard",
        "meaning": "Experimental FEMA-equivalent annual hazard estimate from physical inputs only.",
        "createdAt": report["createdAt"],
        "sourceTable": report["sourceTable"],
        "featureNames": names,
        "numericFeatures": NUMERIC_FEATURES,
        "categoricalFeatures": {CATEGORICAL_FEATURE: categories},
        "monotonicDirections": report["features"]["monotonicDirections"],
        "models": {
            "onePercent": {
                "target": TARGETS["onePercent"],
                "path": str(ONE_PERCENT_MODEL_PATH.relative_to(ROOT)),
                "selected": selected["onePercent"],
            },
            "atLeastModerate": {
                "target": TARGETS["atLeastModerate"],
                "path": str(MODERATE_MODEL_PATH.relative_to(ROOT)),
                "selected": selected["atLeastModerate"],
            },
        },
        "evaluationReport": str(REPORT_JSON_PATH.relative_to(ROOT)),
        "productionStatus": "exported_for_app",
    }

    production_artifact = {
        "schemaVersion": 3,
        "modelType": "ordered_gradient_boosted_physical_hazard",
        "meaning": "Experimental FEMA-equivalent annual hazard estimate from physical inputs only.",
        "createdAt": report["createdAt"],
        "sourceTable": report["sourceTable"],
        "training": {
            "datasetRows": len(rows),
            "rows": len(model_rows),
            "testRows": len(splits["test"]),
            "classCounts": report["classCounts"],
            "spatialFoldDegrees": SPATIAL_BLOCK_DEGREES,
            "split": report["split"],
            "library": f"xgboost {xgb.__version__}",
        },
        "annualChanceAnchors": ANNUAL_CHANCE_ANCHORS,
        "minimalAnchorAssumption": (
            "The central estimate uses 0.1% annual chance, the midpoint of the "
            "0%-0.2% interval for minimal-hazard areas. It is a modeling "
            "assumption, not a FEMA probability."
        ),
        "numericFeatures": NUMERIC_FEATURES,
        "numericRanges": numeric_ranges(model_rows),
        "categoricalFeatures": {CATEGORICAL_FEATURE: categories},
        "featureNames": names,
        "monotonicDirections": report["features"]["monotonicDirections"],
        "terrainConfiguration": {
            "sourceResolutionMeters": 10,
            "runtimeResolutionMeters": 30,
            "runtimeArtifact": "terrain-features-30m.tif",
        },
        "models": {
            "onePercent": compact_booster(one_percent_model),
            "atLeastModerate": compact_booster(moderate_model),
        },
        "spatialValidation": {
            "method": "untouched geographic-block test set after validation selection",
            "meanOnePercentAuc": boosted_test["onePercent"]["rocAuc"],
            "meanAtLeastModerateAuc": boosted_test["atLeastModerate"]["rocAuc"],
            "meanOnePercentBrier": boosted_test["onePercent"]["brierScore"],
            "meanAtLeastModerateBrier": boosted_test["atLeastModerate"]["brierScore"],
            "meanClassAccuracy": boosted_test["classification"]["accuracy"],
            "classification": boosted_test["classification"],
        },
        "output": {
            "horizonYears": 100,
            "conversion": "1 - (1 - annualChance)^100",
        },
    }

    for directory in (ONE_PERCENT_MODEL_PATH.parent, REPORT_JSON_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
    one_percent_model.save_model(ONE_PERCENT_MODEL_PATH)
    moderate_model.save_model(MODERATE_MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    PRODUCTION_MODEL_PATH.write_text(
        json.dumps(production_artifact, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT_MARKDOWN_PATH.write_text(markdown_report(report), encoding="utf-8")
    write_parity_fixture(
        splits["test"],
        matrices["test"],
        names,
        one_percent_model,
        moderate_model,
    )

    print(
        "Untouched geographic test:",
        {
            "previousLogisticBaselineAccuracy": round(
                logistic_test["classification"]["accuracy"], 4
            ),
            "boostedAccuracy": round(boosted_test["classification"]["accuracy"], 4),
            "onePercentAuc": round(boosted_test["onePercent"]["rocAuc"], 4),
            "atLeastModerateAuc": round(
                boosted_test["atLeastModerate"]["rocAuc"], 4
            ),
        },
        flush=True,
    )
    print(f"Recommendation: {recommendation}", flush=True)
    print(f"Wrote {PRODUCTION_MODEL_PATH.relative_to(ROOT)}", flush=True)
    print(f"Wrote {REPORT_MARKDOWN_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
