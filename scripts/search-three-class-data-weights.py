#!/usr/bin/env python3
"""Search added-data fractions and class-weight strengths without test leakage."""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "physical-hazard-calibration.csv"
EXPANDED_PATH = ROOT / "data" / "physical-hazard-calibration-expanded.csv"
MODEL_PATH = ROOT / "models" / "flood-physical-hazard-three-class-search-winner.json"
METADATA_PATH = ROOT / "models" / "flood-physical-hazard-three-class-search-winner-metadata.json"
REPORT_JSON_PATH = ROOT / "reports" / "three-class-data-weight-search.json"
REPORT_MARKDOWN_PATH = ROOT / "reports" / "three-class-data-weight-search.md"
PREVIOUS_REPORT_PATH = ROOT / "reports" / "three-class-model-evaluation.json"

DATA_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
WEIGHT_STRENGTHS = [0.0, 0.35, 0.7, 1.0]
MIDDLE_RECALL_FLOOR = 0.30
ACCURACY_FLOOR = 0.80


def load_shared_module() -> Any:
    path = ROOT / "scripts" / "train-three-class-hazard-benchmark.py"
    specification = importlib.util.spec_from_file_location(
        "three_class_benchmark_shared", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {path.relative_to(ROOT)}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


shared = load_shared_module()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def is_added_row(row: dict[str, str]) -> bool:
    return "-extra-" in row["sample_id"]


def deterministic_added_order(row: dict[str, str]) -> tuple[int, str]:
    return shared.fnv1a_32(f"search:{row['sample_id']}"), row["sample_id"]


def rows_for_fraction(
    original_training: list[dict[str, str]],
    added_training: list[dict[str, str]],
    fraction: float,
) -> tuple[list[dict[str, str]], int]:
    count = round(len(added_training) * fraction)
    return [*original_training, *added_training[:count]], count


def strength_weights(
    labels: np.ndarray[Any, Any], strength: float
) -> tuple[np.ndarray[Any, Any], dict[str, float]]:
    counts = np.bincount(labels, minlength=len(shared.CLASS_ORDER))
    fully_balanced = labels.size / (len(shared.CLASS_ORDER) * counts)
    by_index = fully_balanced**strength
    by_index /= np.mean(by_index[labels])
    return by_index[labels].astype(np.float32), {
        label: float(by_index[index])
        for index, label in enumerate(shared.CLASS_ORDER)
    }


def train_search_candidate(
    hyperparameters: dict[str, float | int],
    training_matrix: np.ndarray[Any, Any],
    training_labels: np.ndarray[Any, Any],
    training_weights: np.ndarray[Any, Any],
    tuning_data: xgb.DMatrix,
    names: list[str],
) -> tuple[xgb.Booster, int]:
    training_data = xgb.DMatrix(
        training_matrix,
        label=training_labels,
        weight=training_weights,
        feature_names=names,
    )
    model = xgb.train(
        shared.base_parameters(hyperparameters),
        training_data,
        num_boost_round=shared.MAXIMUM_ROUNDS,
        evals=[(tuning_data, "tuning")],
        early_stopping_rounds=shared.EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    return model, model.best_iteration + 1


def selection_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    metrics = entry["tuning"]
    middle = metrics["perClass"]["point_two_percent"]
    meets_floors = (
        middle["recall"] >= MIDDLE_RECALL_FLOOR
        and metrics["accuracy"] >= ACCURACY_FLOOR
    )
    return (
        not meets_floors,
        -metrics["macroF1"],
        -middle["f1"],
        metrics["logLoss"],
        metrics["multiclassBrier"],
    )


def train_final(
    selected: dict[str, Any],
    rows: list[dict[str, str]],
    categories: list[str],
    names: list[str],
) -> tuple[xgb.Booster, dict[str, float]]:
    matrix = shared.vectorize(rows, categories)
    labels = shared.labels_for(rows)
    weights, weights_by_class = strength_weights(
        labels, selected["weightStrength"]
    )
    data = xgb.DMatrix(
        matrix, label=labels, weight=weights, feature_names=names
    )
    model = xgb.train(
        shared.base_parameters(selected["hyperparameters"]),
        data,
        num_boost_round=selected["rounds"],
        verbose_eval=False,
    )
    return model, weights_by_class


def markdown_report(report: dict[str, Any]) -> str:
    winner = report["winner"]
    test = winner["testCalibrated"]
    middle = test["perClass"]["point_two_percent"]
    ordered = report["orderedProductionBaseline"]
    previous = report.get("previousThreeClassBenchmark")
    lines = [
        "# Three-class data amount and class-weight search",
        "",
        "All data-fraction, weight-strength, and tree candidates were selected on original tuning blocks. Added validation rows were excluded. Vector scaling was fitted once on original calibration blocks, and the untouched test set was evaluated once after selection.",
        "",
        "## Selected configuration",
        "",
        f"- Added training data: {winner['dataFraction']:.0%} ({winner['addedTrainingRows']} rows)",
        f"- Class-weight strength: {winner['weightStrength']:.2f} (0 is none; 1 is fully balanced)",
        f"- Trees: {winner['rounds']}",
        f"- Hyperparameters: `{json.dumps(winner['hyperparameters'], sort_keys=True)}`",
        "",
        "## Untouched geographic test",
        "",
        "| Metric | Ordered production | Search winner |",
        "| --- | ---: | ---: |",
        f"| Accuracy | {ordered['accuracy']:.2%} | {test['accuracy']:.2%} |",
        f"| Balanced accuracy | {ordered['balancedAccuracy']:.2%} | {test['balancedAccuracy']:.2%} |",
        f"| Macro F1 | {ordered['macroF1']:.4f} | {test['macroF1']:.4f} |",
        f"| 0.2% recall | {ordered['perClass']['point_two_percent']['recall']:.2%} | {middle['recall']:.2%} |",
        f"| 0.2% precision | {ordered['perClass']['point_two_percent']['precision']:.2%} | {middle['precision']:.2%} |",
        f"| Log loss | {ordered['logLoss']:.4f} | {test['logLoss']:.4f} |",
        f"| Multiclass Brier | {ordered['multiclassBrier']:.4f} | {test['multiclassBrier']:.4f} |",
        f"| Calibration error | {ordered['expectedCalibrationError']:.4f} | {test['expectedCalibrationError']:.4f} |",
    ]
    if previous:
        lines.extend(
            [
                "",
                f"The earlier calibrated three-class benchmark had {previous['accuracy']:.2%} accuracy and {previous['perClass']['point_two_percent']['recall']:.2%} 0.2% recall. It was not used to select this search winner.",
            ]
        )
    lines.extend(
        [
            "",
            "The production app model was not changed. This remains a FEMA-equivalent physical-hazard benchmark, not a parcel flood-outcome model.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    original_rows = read_rows(BASE_PATH)
    expanded_rows = read_rows(EXPANDED_PATH)
    original_ids = {row["sample_id"] for row in original_rows}
    if len(original_ids) != len(original_rows):
        raise RuntimeError("Original sample IDs are not unique")
    if not original_ids.issubset({row["sample_id"] for row in expanded_rows}):
        raise RuntimeError("Expanded table does not contain every original row")

    original_training = [
        row for row in original_rows if shared.original_partition(row) == "training"
    ]
    original_validation = [
        row for row in original_rows if shared.original_partition(row) == "validation"
    ]
    test_rows = [
        row for row in original_rows if shared.original_partition(row) == "test"
    ]
    tuning_rows, calibration_rows = shared.split_validation_blocks(
        original_validation
    )
    added_training = sorted(
        [
            row
            for row in expanded_rows
            if is_added_row(row) and shared.original_partition(row) == "training"
        ],
        key=deterministic_added_order,
    )
    added_nontraining = [
        row
        for row in expanded_rows
        if is_added_row(row) and shared.original_partition(row) != "training"
    ]
    if any(shared.original_partition(row) == "test" for row in added_nontraining):
        raise RuntimeError("Added data leaked into untouched test blocks")

    categories = shared.categories_for([*original_training, *added_training])
    names = shared.feature_names(categories)
    tuning_matrix = shared.vectorize(tuning_rows, categories)
    tuning_labels = shared.labels_for(tuning_rows)
    tuning_data = xgb.DMatrix(
        tuning_matrix, label=tuning_labels, feature_names=names
    )

    leaderboard: list[dict[str, Any]] = []
    total_configurations = (
        len(DATA_FRACTIONS)
        * len(WEIGHT_STRENGTHS)
        * len(shared.HYPERPARAMETER_GRID)
    )
    completed = 0
    for data_fraction in DATA_FRACTIONS:
        training_rows, added_count = rows_for_fraction(
            original_training, added_training, data_fraction
        )
        training_matrix = shared.vectorize(training_rows, categories)
        training_labels = shared.labels_for(training_rows)
        for weight_strength in WEIGHT_STRENGTHS:
            weights, weights_by_class = strength_weights(
                training_labels, weight_strength
            )
            combination_entries: list[dict[str, Any]] = []
            for hyperparameter_index, hyperparameters in enumerate(
                shared.HYPERPARAMETER_GRID, start=1
            ):
                model, rounds = train_search_candidate(
                    hyperparameters,
                    training_matrix,
                    training_labels,
                    weights,
                    tuning_data,
                    names,
                )
                probabilities = model.predict(
                    tuning_data, iteration_range=(0, rounds)
                )
                metrics = shared.probability_metrics(
                    tuning_labels, probabilities
                )
                entry = {
                    "dataFraction": data_fraction,
                    "addedTrainingRows": added_count,
                    "weightStrength": weight_strength,
                    "classWeights": weights_by_class,
                    "hyperparameterCandidate": hyperparameter_index,
                    "hyperparameters": hyperparameters,
                    "rounds": rounds,
                    "tuning": metrics,
                }
                leaderboard.append(entry)
                combination_entries.append(entry)
                completed += 1
            best_combination = min(combination_entries, key=selection_key)
            middle = best_combination["tuning"]["perClass"][
                "point_two_percent"
            ]
            print(
                f"Search {completed}/{total_configurations}: data={data_fraction:.0%}, "
                f"weight={weight_strength:.2f}, best macro F1="
                f"{best_combination['tuning']['macroF1']:.4f}, accuracy="
                f"{best_combination['tuning']['accuracy']:.3%}, middle recall="
                f"{middle['recall']:.3%}",
                flush=True,
            )

    leaderboard.sort(key=selection_key)
    selected = leaderboard[0]
    selected_training_rows, selected_added_count = rows_for_fraction(
        original_training, added_training, selected["dataFraction"]
    )
    final_rows = [*selected_training_rows, *tuning_rows]
    model, final_weights = train_final(selected, final_rows, categories, names)

    calibration_matrix = shared.vectorize(calibration_rows, categories)
    calibration_labels = shared.labels_for(calibration_rows)
    calibration_data = xgb.DMatrix(calibration_matrix, feature_names=names)
    calibration_margins = model.predict(calibration_data, output_margin=True)
    calibration_parameters, calibration = shared.fit_vector_scaling(
        calibration_margins, calibration_labels
    )
    calibration["before"] = shared.probability_metrics(
        calibration_labels, shared.softmax(calibration_margins)
    )
    calibration["after"] = shared.probability_metrics(
        calibration_labels,
        shared.apply_vector_scaling(calibration_margins, calibration_parameters),
    )

    # The untouched test set is intentionally not vectorized or predicted until
    # every data, weighting, tree, and calibration choice is frozen above.
    test_matrix = shared.vectorize(test_rows, categories)
    test_labels = shared.labels_for(test_rows)
    test_data = xgb.DMatrix(test_matrix, feature_names=names)
    test_margins = model.predict(test_data, output_margin=True)
    test_uncalibrated = shared.probability_metrics(
        test_labels, shared.softmax(test_margins)
    )
    test_calibrated = shared.probability_metrics(
        test_labels,
        shared.apply_vector_scaling(test_margins, calibration_parameters),
    )
    ordered_test = shared.probability_metrics(
        test_labels, shared.load_ordered_probabilities(test_matrix, names)
    )

    previous = None
    if PREVIOUS_REPORT_PATH.exists():
        with PREVIOUS_REPORT_PATH.open(encoding="utf-8") as source:
            previous_report = json.load(source)
        previous = previous_report["threeClass"]["testCalibrated"]

    winner = {
        **{key: value for key, value in selected.items() if key != "tuning"},
        "addedTrainingRows": selected_added_count,
        "finalClassWeights": final_weights,
        "tuning": selected["tuning"],
        "calibration": calibration,
        "testUncalibrated": test_uncalibrated,
        "testCalibrated": test_calibrated,
    }
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "Data-fraction and class-weight search with one untouched test evaluation",
        "productionModelChanged": False,
        "sourceTables": {
            "original": str(BASE_PATH.relative_to(ROOT)),
            "expanded": str(EXPANDED_PATH.relative_to(ROOT)),
        },
        "searchSpace": {
            "dataFractions": DATA_FRACTIONS,
            "weightStrengths": WEIGHT_STRENGTHS,
            "hyperparameterCandidates": shared.HYPERPARAMETER_GRID,
            "configurations": total_configurations,
            "selectionRule": (
                f"require tuning 0.2% recall >= {MIDDLE_RECALL_FLOOR:.0%} and "
                f"accuracy >= {ACCURACY_FLOOR:.0%}, then maximize macro F1 and "
                "0.2% F1, then minimize log loss and Brier score"
            ),
        },
        "dataIsolation": {
            "originalTrainingRows": len(original_training),
            "availableAddedTrainingRows": len(added_training),
            "excludedAddedValidationRows": len(added_nontraining),
            "originalTuningRows": len(tuning_rows),
            "originalCalibrationRows": len(calibration_rows),
            "untouchedTestRows": len(test_rows),
            "testEvaluationsAfterSelection": 1,
        },
        "winner": winner,
        "orderedProductionBaseline": ordered_test,
        "previousThreeClassBenchmark": previous,
        "leaderboard": leaderboard,
        "importantLimit": (
            "These are FEMA-equivalent physical hazard classifications, not "
            "observed 100-year flood outcomes at individual properties."
        ),
    }
    metadata = {
        "schemaVersion": 1,
        "modelType": "three_class_data_weight_search_winner",
        "createdAt": report["createdAt"],
        "classOrder": shared.CLASS_ORDER,
        "featureNames": names,
        "numericFeatures": shared.NUMERIC_FEATURES,
        "categoricalFeatures": {shared.CATEGORICAL_FEATURE: categories},
        "selected": {
            key: winner[key]
            for key in (
                "dataFraction",
                "addedTrainingRows",
                "weightStrength",
                "hyperparameters",
                "rounds",
                "finalClassWeights",
            )
        },
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
    print(
        "Selected configuration:",
        {
            "dataFraction": selected["dataFraction"],
            "addedTrainingRows": selected_added_count,
            "weightStrength": selected["weightStrength"],
            "hyperparameterCandidate": selected["hyperparameterCandidate"],
            "rounds": selected["rounds"],
        },
        flush=True,
    )
    print(
        "Untouched geographic test (evaluated once after selection):",
        {
            "accuracy": round(test_calibrated["accuracy"], 4),
            "middleRecall": round(middle["recall"], 4),
            "middlePrecision": round(middle["precision"], 4),
            "macroF1": round(test_calibrated["macroF1"], 4),
            "logLoss": round(test_calibrated["logLoss"], 4),
            "multiclassBrier": round(test_calibrated["multiclassBrier"], 4),
        },
        flush=True,
    )
    print("Production model unchanged.", flush=True)
    print(f"Wrote {REPORT_MARKDOWN_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
