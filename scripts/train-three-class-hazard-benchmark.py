#!/usr/bin/env python3
"""Benchmark a balanced, calibrated, direct three-class XGBoost model.

The production ordered model is never modified. Geographic blocks are divided
into training, tuning, calibration, and untouched test sets. Hyperparameters
are selected on tuning blocks, vector scaling is fitted on separate calibration
blocks, and final metrics are computed once on the original test blocks.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_ARGUMENT = next(
    (argument[8:] for argument in sys.argv[1:] if argument.startswith("--input=")),
    "data/physical-hazard-calibration.csv",
)
TAG_ARGUMENT = next(
    (argument[6:] for argument in sys.argv[1:] if argument.startswith("--tag=")), ""
)
if TAG_ARGUMENT and not re.fullmatch(r"[a-z0-9-]+", TAG_ARGUMENT):
    raise ValueError("--tag may contain only lowercase letters, numbers, and hyphens")
ARTIFACT_SUFFIX = f"-{TAG_ARGUMENT}" if TAG_ARGUMENT else ""
INPUT_PATH = ROOT / INPUT_ARGUMENT
ORDERED_METADATA_PATH = ROOT / "models" / "flood-physical-hazard-boosted-metadata.json"
MODEL_PATH = ROOT / "models" / f"flood-physical-hazard-three-class{ARTIFACT_SUFFIX}-benchmark.json"
METADATA_PATH = ROOT / "models" / f"flood-physical-hazard-three-class{ARTIFACT_SUFFIX}-metadata.json"
REPORT_JSON_PATH = ROOT / "reports" / f"three-class-model-evaluation{ARTIFACT_SUFFIX}.json"
REPORT_MARKDOWN_PATH = ROOT / "reports" / f"three-class-model-evaluation{ARTIFACT_SUFFIX}.md"

SPATIAL_BLOCK_DEGREES = 0.05
CLASS_ORDER = ["one_percent", "point_two_percent", "minimal"]
CLASS_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}
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
HYPERPARAMETER_GRID = [
    {"max_depth": 2, "eta": 0.03, "min_child_weight": 5, "reg_lambda": 5},
    {"max_depth": 3, "eta": 0.03, "min_child_weight": 10, "reg_lambda": 5},
    {"max_depth": 3, "eta": 0.05, "min_child_weight": 10, "reg_lambda": 10},
    {"max_depth": 3, "eta": 0.03, "min_child_weight": 20, "reg_lambda": 10},
    {"max_depth": 4, "eta": 0.03, "min_child_weight": 20, "reg_lambda": 10},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 20, "reg_lambda": 15},
]
MAXIMUM_ROUNDS = 2_000
EARLY_STOPPING_ROUNDS = 75
SEED = 42
CALIBRATION_L2 = 0.001


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
    return math.log1p(max(0.0, value)) if feature.startswith("log_") else value


def fnv1a_32(value: str) -> int:
    hash_value = 2_166_136_261
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 16_777_619) & 0xFFFFFFFF
    return hash_value


def block_key(row: dict[str, str]) -> str:
    latitude_tile = math.floor(float(row["lat"]) / SPATIAL_BLOCK_DEGREES)
    longitude_tile = math.floor(float(row["lng"]) / SPATIAL_BLOCK_DEGREES)
    return f"{latitude_tile},{longitude_tile}"


def original_partition(row: dict[str, str]) -> str:
    bucket = fnv1a_32(block_key(row)) % 20
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
            if row["mapped_hazard_class"] in CLASS_INDEX
        ]
    if len(rows) < 1_000:
        raise RuntimeError(f"{INPUT_PATH} needs at least 1,000 valid rows")
    return rows


def row_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts = Counter(row["mapped_hazard_class"] for row in rows)
    counts["rows"] = len(rows)
    return counts


def split_validation_blocks(
    validation_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Stratify whole validation blocks into tuning and calibration subsets."""
    blocks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in validation_rows:
        blocks[block_key(row)].append(row)
    total = row_counts(validation_rows)
    assignments: dict[str, list[dict[str, str]]] = {
        "tuning": [],
        "calibration": [],
    }
    assigned_counts = {name: Counter() for name in assignments}
    ordered_blocks = sorted(
        blocks.items(),
        key=lambda item: (
            -row_counts(item[1])["point_two_percent"],
            -row_counts(item[1])["one_percent"],
            -len(item[1]),
            item[0],
        ),
    )
    fields = ["rows", *CLASS_ORDER]
    for _, rows in ordered_blocks:
        candidate_counts = row_counts(rows)

        def allocation_score(chosen: str) -> float:
            score = 0.0
            for name in assignments:
                counts = assigned_counts[name]
                for field in fields:
                    value = counts[field] + (
                        candidate_counts[field] if name == chosen else 0
                    )
                    score += ((value - total[field] / 2) / max(1, total[field])) ** 2
            return score

        chosen = min(assignments, key=lambda name: (allocation_score(name), name))
        assignments[chosen].extend(rows)
        assigned_counts[chosen].update(candidate_counts)
    return assignments["tuning"], assignments["calibration"]


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
    matrix = np.zeros(
        (len(rows), len(NUMERIC_FEATURES) + len(categories)), dtype=np.float32
    )
    for row_index, row in enumerate(rows):
        for feature_index, feature in enumerate(NUMERIC_FEATURES):
            matrix[row_index, feature_index] = numeric_value(row, feature)
        category = row.get(CATEGORICAL_FEATURE) or "Unknown"
        category_index = category_indexes.get(category)
        if category_index is not None:
            matrix[row_index, len(NUMERIC_FEATURES) + category_index] = 1.0
    return matrix


def labels_for(rows: list[dict[str, str]]) -> np.ndarray[Any, np.dtype[np.int64]]:
    return np.asarray(
        [CLASS_INDEX[row["mapped_hazard_class"]] for row in rows], dtype=np.int64
    )


def balanced_weights(labels: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], dict[str, float]]:
    counts = np.bincount(labels, minlength=len(CLASS_ORDER))
    weights_by_index = labels.size / (len(CLASS_ORDER) * counts)
    weights = weights_by_index[labels].astype(np.float32)
    return weights, {
        label: float(weights_by_index[index])
        for index, label in enumerate(CLASS_ORDER)
    }


def base_parameters(candidate: dict[str, float | int]) -> dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": len(CLASS_ORDER),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_bin": 512,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": SEED,
        "nthread": 0,
        **candidate,
    }


def softmax(margins: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    shifted = margins - np.max(margins, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def expected_calibration_error(
    actual: np.ndarray[Any, Any], probabilities: np.ndarray[Any, Any], bins: int = 10
) -> float:
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predicted == actual
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return error


def probability_metrics(
    actual: np.ndarray[Any, Any], probabilities: np.ndarray[Any, Any]
) -> dict[str, Any]:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = np.argmax(probabilities, axis=1)
    actual_names = [CLASS_ORDER[index] for index in actual]
    predicted_names = [CLASS_ORDER[index] for index in predicted]
    report = classification_report(
        actual_names,
        predicted_names,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    one_hot = np.eye(len(CLASS_ORDER))[actual]
    per_class_auc = {
        label: float(roc_auc_score(one_hot[:, index], probabilities[:, index]))
        for index, label in enumerate(CLASS_ORDER)
    }
    return {
        "rows": int(actual.size),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balancedAccuracy": float(balanced_accuracy_score(actual, predicted)),
        "macroF1": float(f1_score(actual, predicted, average="macro")),
        "logLoss": float(log_loss(actual, probabilities, labels=range(len(CLASS_ORDER)))),
        "multiclassBrier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expectedCalibrationError": expected_calibration_error(actual, probabilities),
        "macroOneVsRestRocAuc": float(
            roc_auc_score(actual, probabilities, multi_class="ovr", average="macro")
        ),
        "perClassRocAuc": per_class_auc,
        "confusionMatrix": confusion_matrix(
            actual_names, predicted_names, labels=CLASS_ORDER
        ).tolist(),
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


def train_candidate(
    candidate: dict[str, float | int],
    training_matrix: np.ndarray[Any, Any],
    training_labels: np.ndarray[Any, Any],
    tuning_matrix: np.ndarray[Any, Any],
    tuning_labels: np.ndarray[Any, Any],
    names: list[str],
) -> tuple[xgb.Booster, int]:
    weights, _ = balanced_weights(training_labels)
    training_data = xgb.DMatrix(
        training_matrix,
        label=training_labels,
        weight=weights,
        feature_names=names,
    )
    tuning_data = xgb.DMatrix(
        tuning_matrix, label=tuning_labels, feature_names=names
    )
    model = xgb.train(
        base_parameters(candidate),
        training_data,
        num_boost_round=MAXIMUM_ROUNDS,
        evals=[(tuning_data, "tuning")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    return model, model.best_iteration + 1


def tune_model(
    training_matrix: np.ndarray[Any, Any],
    training_labels: np.ndarray[Any, Any],
    tuning_matrix: np.ndarray[Any, Any],
    tuning_labels: np.ndarray[Any, Any],
    names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tuning_data = xgb.DMatrix(tuning_matrix, feature_names=names)
    leaderboard: list[dict[str, Any]] = []
    for index, candidate in enumerate(HYPERPARAMETER_GRID, start=1):
        model, rounds = train_candidate(
            candidate,
            training_matrix,
            training_labels,
            tuning_matrix,
            tuning_labels,
            names,
        )
        probabilities = model.predict(tuning_data, iteration_range=(0, rounds))
        metrics = probability_metrics(tuning_labels, probabilities)
        entry = {
            "candidate": index,
            "hyperparameters": candidate,
            "rounds": rounds,
            "tuning": metrics,
        }
        leaderboard.append(entry)
        middle = metrics["perClass"]["point_two_percent"]
        print(
            f"  candidate {index}/{len(HYPERPARAMETER_GRID)}: rounds={rounds}, "
            f"macro F1={metrics['macroF1']:.4f}, middle recall={middle['recall']:.3%}, "
            f"middle F1={middle['f1']:.4f}, log loss={metrics['logLoss']:.4f}",
            flush=True,
        )
    leaderboard.sort(
        key=lambda entry: (
            -entry["tuning"]["macroF1"],
            -entry["tuning"]["perClass"]["point_two_percent"]["f1"],
            entry["tuning"]["logLoss"],
        )
    )
    return leaderboard[0], leaderboard


def train_final_model(
    selected: dict[str, Any],
    model_matrix: np.ndarray[Any, Any],
    model_labels: np.ndarray[Any, Any],
    names: list[str],
) -> tuple[xgb.Booster, dict[str, float]]:
    weights, weights_by_class = balanced_weights(model_labels)
    data = xgb.DMatrix(
        model_matrix, label=model_labels, weight=weights, feature_names=names
    )
    model = xgb.train(
        base_parameters(selected["hyperparameters"]),
        data,
        num_boost_round=selected["rounds"],
        verbose_eval=False,
    )
    return model, weights_by_class


def apply_vector_scaling(
    margins: np.ndarray[Any, Any], parameters: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    scales = np.exp(parameters[: len(CLASS_ORDER)])
    biases = np.append(parameters[len(CLASS_ORDER) :], 0.0)
    return softmax(margins * scales + biases)


def fit_vector_scaling(
    margins: np.ndarray[Any, Any], actual: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    parameter_count = len(CLASS_ORDER) + len(CLASS_ORDER) - 1

    def objective(parameters: np.ndarray[Any, Any]) -> float:
        probabilities = apply_vector_scaling(margins, parameters)
        negative_log_likelihood = -np.mean(
            np.log(np.clip(probabilities[np.arange(actual.size), actual], 1e-12, 1.0))
        )
        return float(negative_log_likelihood + CALIBRATION_L2 * np.sum(parameters**2))

    initial = np.zeros(parameter_count, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0)] * len(CLASS_ORDER)
        + [(-5.0, 5.0)] * (len(CLASS_ORDER) - 1),
    )
    if not result.success:
        raise RuntimeError(f"Vector scaling failed: {result.message}")
    parameters = np.asarray(result.x, dtype=np.float64)
    return parameters, {
        "method": "vector_scaling",
        "formula": "softmax(rawMargins * exp(logScales) + biases)",
        "logScales": parameters[: len(CLASS_ORDER)].tolist(),
        "scales": np.exp(parameters[: len(CLASS_ORDER)]).tolist(),
        "biases": np.append(parameters[len(CLASS_ORDER) :], 0.0).tolist(),
        "l2Penalty": CALIBRATION_L2,
        "optimizer": "L-BFGS-B",
        "success": bool(result.success),
        "iterations": int(result.nit),
    }


def load_ordered_probabilities(
    matrix: np.ndarray[Any, Any], names: list[str]
) -> np.ndarray[Any, Any]:
    with ORDERED_METADATA_PATH.open(encoding="utf-8") as source:
        metadata = json.load(source)
    expected_names = metadata["featureNames"]
    if expected_names != names:
        raise RuntimeError("The ordered baseline and benchmark feature order differ")
    models = []
    for key in ("onePercent", "atLeastModerate"):
        model = xgb.Booster()
        model.load_model(ROOT / metadata["models"][key]["path"])
        models.append(model)
    data = xgb.DMatrix(matrix, feature_names=names)
    one_percent = models[0].predict(data)
    at_least_moderate = np.maximum(one_percent, models[1].predict(data))
    return np.column_stack(
        (one_percent, at_least_moderate - one_percent, 1.0 - at_least_moderate)
    )


def split_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = row_counts(rows)
    return {
        "rows": len(rows),
        "geographicBlocks": len({block_key(row) for row in rows}),
        "classCounts": {label: counts[label] for label in CLASS_ORDER},
    }


def markdown_report(report: dict[str, Any]) -> str:
    baseline = report["orderedProductionBaseline"]["test"]
    uncalibrated = report["threeClass"]["testUncalibrated"]
    calibrated = report["threeClass"]["testCalibrated"]

    def percent(value: float) -> str:
        return f"{value:.1%}"

    lines = [
        "# Balanced three-class XGBoost benchmark",
        "",
        "The production model was not changed. Hyperparameters, probability calibration, and final evaluation use separate geographic blocks.",
        "",
        "| Test metric | Ordered production | Three-class uncalibrated | Three-class calibrated |",
        "| --- | ---: | ---: | ---: |",
        f"| Accuracy | {percent(baseline['accuracy'])} | {percent(uncalibrated['accuracy'])} | {percent(calibrated['accuracy'])} |",
        f"| Balanced accuracy | {percent(baseline['balancedAccuracy'])} | {percent(uncalibrated['balancedAccuracy'])} | {percent(calibrated['balancedAccuracy'])} |",
        f"| Macro F1 | {baseline['macroF1']:.4f} | {uncalibrated['macroF1']:.4f} | {calibrated['macroF1']:.4f} |",
        f"| 0.2% recall | {percent(baseline['perClass']['point_two_percent']['recall'])} | {percent(uncalibrated['perClass']['point_two_percent']['recall'])} | {percent(calibrated['perClass']['point_two_percent']['recall'])} |",
        f"| 0.2% precision | {percent(baseline['perClass']['point_two_percent']['precision'])} | {percent(uncalibrated['perClass']['point_two_percent']['precision'])} | {percent(calibrated['perClass']['point_two_percent']['precision'])} |",
        f"| Log loss (lower is better) | {baseline['logLoss']:.4f} | {uncalibrated['logLoss']:.4f} | {calibrated['logLoss']:.4f} |",
        f"| Multiclass Brier (lower is better) | {baseline['multiclassBrier']:.4f} | {uncalibrated['multiclassBrier']:.4f} | {calibrated['multiclassBrier']:.4f} |",
        f"| Calibration error (lower is better) | {baseline['expectedCalibrationError']:.4f} | {uncalibrated['expectedCalibrationError']:.4f} | {calibrated['expectedCalibrationError']:.4f} |",
        "",
        "## Split",
        "",
        "| Purpose | Rows | Geographic blocks | 0.2% rows |",
        "| --- | ---: | ---: | ---: |",
        *(f"| {name.title()} | {summary['rows']} | {summary['geographicBlocks']} | {summary['classCounts']['point_two_percent']} |" for name, summary in report["split"].items()),
        "",
        "The direct model is trained with inverse-frequency class weights. Vector scaling is then fitted on the calibration-only blocks to correct the weighted model's probabilities.",
        "",
        "## Conclusion",
        "",
        f"The calibrated model changes 0.2% recall from {percent(baseline['perClass']['point_two_percent']['recall'])} to {percent(calibrated['perClass']['point_two_percent']['recall'])}, while changing overall accuracy from {percent(baseline['accuracy'])} to {percent(calibrated['accuracy'])}. Log loss and multiclass Brier score also improve. Expected calibration error is slightly worse on the test set, so it remains explicitly reported rather than treated as a universal calibration win.",
        "",
        "Recommendation: advance this benchmark to physical-behavior and stability audits before considering production integration. Direct multiclass trees do not currently carry the ordered model's monotonic guarantees.",
        "",
        "Important: these remain FEMA-equivalent hazard-class probabilities, not observed parcel flood outcomes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    training_rows = [row for row in rows if original_partition(row) == "training"]
    validation_rows = [row for row in rows if original_partition(row) == "validation"]
    test_rows = [row for row in rows if original_partition(row) == "test"]
    tuning_rows, calibration_rows = split_validation_blocks(validation_rows)
    splits = {
        "training": training_rows,
        "tuning": tuning_rows,
        "calibration": calibration_rows,
        "test": test_rows,
    }
    block_sets = {
        name: {block_key(row) for row in split_rows}
        for name, split_rows in splits.items()
    }
    for first_name, first_blocks in block_sets.items():
        for second_name, second_blocks in block_sets.items():
            if first_name < second_name and first_blocks & second_blocks:
                raise RuntimeError(f"Geographic leakage between {first_name} and {second_name}")
    print("Geographic split:", {name: split_summary(value) for name, value in splits.items()}, flush=True)

    categories = categories_for(training_rows)
    names = feature_names(categories)
    matrices = {name: vectorize(split_rows, categories) for name, split_rows in splits.items()}
    labels = {name: labels_for(split_rows) for name, split_rows in splits.items()}

    print("Tuning balanced three-class models...", flush=True)
    selected, leaderboard = tune_model(
        matrices["training"],
        labels["training"],
        matrices["tuning"],
        labels["tuning"],
        names,
    )
    model_rows = [*training_rows, *tuning_rows]
    model_matrix = vectorize(model_rows, categories)
    model_labels = labels_for(model_rows)
    model, class_weights = train_final_model(
        selected, model_matrix, model_labels, names
    )

    calibration_data = xgb.DMatrix(matrices["calibration"], feature_names=names)
    calibration_margins = model.predict(calibration_data, output_margin=True)
    calibration_parameters, calibration = fit_vector_scaling(
        calibration_margins, labels["calibration"]
    )
    calibration["before"] = probability_metrics(
        labels["calibration"], softmax(calibration_margins)
    )
    calibration["after"] = probability_metrics(
        labels["calibration"],
        apply_vector_scaling(calibration_margins, calibration_parameters),
    )

    test_data = xgb.DMatrix(matrices["test"], feature_names=names)
    test_margins = model.predict(test_data, output_margin=True)
    test_uncalibrated = probability_metrics(labels["test"], softmax(test_margins))
    test_calibrated = probability_metrics(
        labels["test"], apply_vector_scaling(test_margins, calibration_parameters)
    )
    ordered_test = probability_metrics(
        labels["test"], load_ordered_probabilities(matrices["test"], names)
    )
    comparison = {
        "accuracyChange": test_calibrated["accuracy"] - ordered_test["accuracy"],
        "balancedAccuracyChange": (
            test_calibrated["balancedAccuracy"] - ordered_test["balancedAccuracy"]
        ),
        "macroF1Change": test_calibrated["macroF1"] - ordered_test["macroF1"],
        "pointTwoRecallChange": (
            test_calibrated["perClass"]["point_two_percent"]["recall"]
            - ordered_test["perClass"]["point_two_percent"]["recall"]
        ),
        "logLossChange": test_calibrated["logLoss"] - ordered_test["logLoss"],
        "multiclassBrierChange": (
            test_calibrated["multiclassBrier"] - ordered_test["multiclassBrier"]
        ),
        "expectedCalibrationErrorChange": (
            test_calibrated["expectedCalibrationError"]
            - ordered_test["expectedCalibrationError"]
        ),
        "passesBenchmarkGate": (
            test_calibrated["accuracy"] > ordered_test["accuracy"]
            and test_calibrated["macroF1"] > ordered_test["macroF1"]
            and test_calibrated["perClass"]["point_two_percent"]["recall"]
            > ordered_test["perClass"]["point_two_percent"]["recall"]
            and test_calibrated["logLoss"] < ordered_test["logLoss"]
            and test_calibrated["multiclassBrier"]
            < ordered_test["multiclassBrier"]
        ),
        "recommendation": (
            "advance to physical-behavior and stability audits before production integration"
        ),
    }

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceTable": str(INPUT_PATH.relative_to(ROOT)),
        "purpose": "Balanced direct three-class benchmark with held-out probability calibration",
        "productionModelChanged": False,
        "split": {name: split_summary(split_rows) for name, split_rows in splits.items()},
        "selection": {
            "criterion": "highest tuning macro F1, then 0.2% F1, then lowest log loss",
            "selected": selected,
            "leaderboard": leaderboard,
        },
        "training": {
            "modelRows": len(model_rows),
            "classWeights": class_weights,
            "classWeightFormula": "rows / (classCount * numberOfClasses)",
            "library": f"xgboost {xgb.__version__}",
            "features": {"numeric": NUMERIC_FEATURES, "categories": categories, "expanded": names},
            "directMulticlassMonotonicConstraintLimitation": (
                "No monotonic constraints are applied because XGBoost applies one direction "
                "per feature to every output group, while safer terrain should decrease high-"
                "hazard probabilities and increase minimal-hazard probability."
            ),
        },
        "probabilityCalibration": calibration,
        "threeClass": {
            "testUncalibrated": test_uncalibrated,
            "testCalibrated": test_calibrated,
        },
        "orderedProductionBaseline": {"test": ordered_test},
        "comparison": comparison,
        "importantLimit": (
            "These are FEMA-equivalent physical hazard classifications, not observed "
            "100-year flood outcomes at individual properties."
        ),
    }
    metadata = {
        "schemaVersion": 1,
        "modelType": "balanced_calibrated_three_class_xgboost_benchmark",
        "createdAt": report["createdAt"],
        "classOrder": CLASS_ORDER,
        "featureNames": names,
        "numericFeatures": NUMERIC_FEATURES,
        "categoricalFeatures": {CATEGORICAL_FEATURE: categories},
        "selected": selected,
        "classWeights": class_weights,
        "calibration": {
            key: calibration[key]
            for key in ("method", "formula", "logScales", "scales", "biases")
        },
        "nativeModel": str(MODEL_PATH.relative_to(ROOT)),
        "evaluationReport": str(REPORT_JSON_PATH.relative_to(ROOT)),
        "productionStatus": "benchmark_only_not_used_by_app",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT_MARKDOWN_PATH.write_text(markdown_report(report), encoding="utf-8")

    middle = test_calibrated["perClass"]["point_two_percent"]
    baseline_middle = ordered_test["perClass"]["point_two_percent"]
    print(
        "Untouched geographic test:",
        {
            "orderedAccuracy": round(ordered_test["accuracy"], 4),
            "threeClassAccuracy": round(test_calibrated["accuracy"], 4),
            "orderedMiddleRecall": round(baseline_middle["recall"], 4),
            "threeClassMiddleRecall": round(middle["recall"], 4),
            "threeClassMiddlePrecision": round(middle["precision"], 4),
            "threeClassLogLoss": round(test_calibrated["logLoss"], 4),
            "threeClassBrier": round(test_calibrated["multiclassBrier"], 4),
        },
        flush=True,
    )
    print("Production model unchanged.", flush=True)
    print(f"Wrote {REPORT_MARKDOWN_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
