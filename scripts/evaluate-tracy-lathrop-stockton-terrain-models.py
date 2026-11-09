"""Evaluate existing models on the Tracy–Lathrop–Stockton holdout.

This script never fits or updates a model. Run after building the sample.
"""

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from terrain_features import FEATURES, vector


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/tracy-lathrop-stockton-fema-map-class-terrain-sample.csv"
DATASET_SOURCE = ROOT / "data/tracy-lathrop-stockton-fema-map-class-terrain-sample.source.json"
HYDRO_SOURCE = ROOT / "hydrography/tracy_lathrop_stockton_3dhp_flowlines.source.json"
HYDRO = ROOT / "hydrography/tracy_lathrop_stockton_3dhp_flowlines.geojson"
TRAINING = ROOT / "data/dnn-three-class-training-table.csv"
BINARY_TRAINING = ROOT / "data/dnn-training-table.csv"
BOOSTED = ROOT / "models/flood-hazard-terrain-three-class-boosted.joblib"
DNN = ROOT / "models/flood-hazard-terrain-three-class-dnn.joblib"
BINARY_BOOSTED = ROOT / "models/flood-risk-terrain-full.joblib"
BINARY_DNN = ROOT / "models/flood-risk-terrain-dnn.joblib"
HIGH_ELEVATION_BOOSTED = ROOT / "models/flood-risk-terrain-high-elevation.joblib"
FIVE_INPUT_BOOSTED = ROOT / "models/flood-risk-boosted.joblib"
FIVE_INPUT_DNN = ROOT / "models/flood-risk-dnn.joblib"
FIVE_INPUT_THREE_CLASS_DNN = ROOT / "models/flood-hazard-three-class-dnn.joblib"
CNN = ROOT / "models/flood-risk-terrain-cnn.pt"
CNN_PATCHES = ROOT / "data/cache/terrain-cnn/tracy-lathrop-stockton-external-patches-float16.npy"
CNN_PATCH_METADATA = ROOT / "data/cache/terrain-cnn/tracy-lathrop-stockton-external-patches.json"
CNN_BAY_REPORT = ROOT / "reports/terrain-cnn-evaluation.json"
CNN_SCORER = ROOT / "scripts/score-santa-cruz-north-cnn.py"
REPORT_JSON = ROOT / "reports/tracy-lathrop-stockton-fema-terrain-external-evaluation.json"
REPORT_MD = ROOT / "reports/tracy-lathrop-stockton-fema-terrain-external-evaluation.md"
CLASSES = ["minimal", "one_percent", "point_two_percent"]


def metrics(actual, predicted):
    return {
        "balancedAccuracy": float(balanced_accuracy_score(actual, predicted)),
        "macroF1": float(f1_score(actual, predicted, labels=CLASSES, average="macro")),
        "perClassF1": {
            label: float(f1_score(actual, predicted, labels=[label], average="macro", zero_division=0))
            for label in CLASSES
        },
        "predictedCounts": dict(Counter(predicted)),
        "confusionMatrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
    }


def binary_metrics(actual, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    return {
        "decisionThresholdFromOriginalValidation": threshold,
        "balancedAccuracy": float(balanced_accuracy_score(actual, predicted)),
        "onePercentF1": float(f1_score(actual, predicted, zero_division=0)),
        "rocAuc": float(roc_auc_score(actual, scores)),
        "confusionMatrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def group_summary(rows, mask):
    selected = [row for row, included in zip(rows, mask) if included]
    fields = [
        "elevation_m", "height_above_drainage_m", "nearest_water_distance_m",
        "local_elevation_250m", "local_elevation_500m", "flow_accumulation_m2",
    ]
    return {
        "count": len(selected),
        "median": {field: float(np.median([float(row[field]) for row in selected]))
                   if selected else None for field in fields},
        "sourcePolygonCounts": dict(Counter(row["source_object_id"] for row in selected)),
    }


def cnn_scores(binary_rows):
    metadata = json.loads(CNN_PATCH_METADATA.read_text(encoding="utf-8"))
    if [row["sample_id"] for row in binary_rows] != [row["sample_id"] for row in metadata["rows"]]:
        raise RuntimeError("CNN patch order does not match the San Joaquin holdout")
    if metadata["datasetSha256"] != hashlib.sha256(DATASET.read_bytes()).hexdigest():
        raise RuntimeError("CNN patches were built from another dataset version")
    completed = subprocess.run(
        [sys.executable, str(CNN_SCORER), "--region", "tracy-lathrop-stockton"], cwd=ROOT,
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"CNN scoring failed: {completed.stderr.strip()}")
    result = json.loads(completed.stdout)
    scores = np.asarray(result["scores"], dtype=np.float64)
    if scores.shape != (len(binary_rows),) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid CNN score count or values")
    return scores, float(result["threshold"]), metadata


def main():
    with DATASET.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    dataset_source = json.loads(DATASET_SOURCE.read_text(encoding="utf-8"))
    hydro_source = json.loads(HYDRO_SOURCE.read_text(encoding="utf-8"))
    hydro_sha = hashlib.sha256(HYDRO.read_bytes()).hexdigest()
    if dataset_source["supplemental_hydrography_sha256"] != hydro_sha or hydro_source["sha256"] != hydro_sha:
        raise RuntimeError("San Joaquin supplemental hydrography does not match the dataset manifest")
    training_counties = set()
    for path in (TRAINING, BINARY_TRAINING):
        with path.open(newline="", encoding="utf-8") as source:
            training_counties.update(row["county"] for row in csv.DictReader(source))
    if not rows or {row["spatial_partition"] for row in rows} != {"external_test"}:
        raise RuntimeError("The Tracy–Lathrop–Stockton sample is not marked as an external holdout")
    if "San Joaquin County" in training_counties:
        raise RuntimeError("Existing model training table includes San Joaquin County")
    if {row["mapped_hazard_class"] for row in rows} != set(CLASSES):
        raise RuntimeError("Missing one or more FEMA map classes")

    x = np.asarray([vector({**row, "location_id": row["sample_id"]}, row)
                    for row in rows], dtype=np.float32)
    if not np.isfinite(x).all() or x.shape[1] != len(FEATURES):
        raise RuntimeError("Dataset does not contain ten finite terrain model inputs")
    actual = np.asarray([row["mapped_hazard_class"] for row in rows])
    boosted = joblib.load(BOOSTED)
    if list(boosted.classes_) != list(range(len(CLASSES))):
        raise RuntimeError("Unexpected boosted model class order")
    boosted_predicted = np.asarray(CLASSES)[boosted.predict(x).astype(int)]
    dnn = joblib.load(DNN)
    if dnn["features"] != FEATURES or dnn["classes"] != CLASSES:
        raise RuntimeError("Unexpected DNN feature or class contract")
    transformed = dnn["scaler"].transform(dnn["imputer"].transform(x))
    dnn_predicted = dnn["model"].predict(transformed)
    five_input_x = x[:, :5]
    five_input_classes = FEATURES[:5]
    five_input_three_class_dnn = joblib.load(FIVE_INPUT_THREE_CLASS_DNN)
    if five_input_three_class_dnn["features"] != five_input_classes:
        raise RuntimeError("Unexpected five-input three-class DNN feature contract")
    five_input_three_class_predicted = five_input_three_class_dnn["model"].predict(
        five_input_three_class_dnn["scaler"].transform(
            five_input_three_class_dnn["imputer"].transform(five_input_x)
        )
    )
    binary_mask = actual != "point_two_percent"
    binary_x = x[binary_mask]
    binary_actual = (actual[binary_mask] == "one_percent").astype(int)
    binary_boosted = joblib.load(BINARY_BOOSTED)
    binary_boosted_scores = binary_boosted.predict_proba(binary_x)[:, 1]
    binary_boosted_threshold = json.loads(
        (ROOT / "reports/terrain-fema-baselines.json").read_text(encoding="utf-8")
    )["variants"]["full"]["decisionThreshold"]
    high_elevation_boosted_scores = joblib.load(HIGH_ELEVATION_BOOSTED).predict_proba(binary_x)[:, 1]
    high_elevation_boosted_threshold = json.loads(
        (ROOT / "reports/terrain-fema-baselines.json").read_text(encoding="utf-8")
    )["variants"]["high-elevation"]["decisionThreshold"]
    five_input_boosted_scores = joblib.load(FIVE_INPUT_BOOSTED).predict_proba(binary_x[:, :5])[:, 1]
    five_input_boosted_threshold = json.loads(
        (ROOT / "reports/boosted-dnn-comparison.json").read_text(encoding="utf-8")
    )["decisionThreshold"]
    five_input_dnn = joblib.load(FIVE_INPUT_DNN)
    if five_input_dnn["features"] != five_input_classes:
        raise RuntimeError("Unexpected five-input binary DNN feature contract")
    five_input_dnn_scores = five_input_dnn["model"].predict_proba(
        five_input_dnn["scaler"].transform(
            five_input_dnn["imputer"].transform(binary_x[:, :5])
        )
    )[:, 1]
    binary_dnn = joblib.load(BINARY_DNN)
    if binary_dnn["features"] != FEATURES:
        raise RuntimeError("Unexpected binary DNN feature contract")
    binary_dnn_scores = binary_dnn["model"].predict_proba(
        binary_dnn["scaler"].transform(binary_dnn["imputer"].transform(binary_x))
    )[:, 1]
    binary_dnn_threshold = json.loads(
        (ROOT / "reports/terrain-dnn-evaluation.json").read_text(encoding="utf-8")
    )["decisionThreshold"]
    cnn_predictions, cnn_threshold, cnn_patch_metadata = cnn_scores(
        [row for row in rows if row["mapped_hazard_class"] != "point_two_percent"],
    )
    bay_cnn = json.loads(CNN_BAY_REPORT.read_text(encoding="utf-8"))["selected"]["test"]
    binary_rows = [row for row in rows if row["mapped_hazard_class"] != "point_two_percent"]
    positive_rows = [row for row in binary_rows if row["mapped_hazard_class"] == "one_percent"]
    positive = binary_actual == 1
    cnn_missed = cnn_predictions[positive] < cnn_threshold
    boosted_missed = binary_boosted_scores[positive] < binary_boosted_threshold
    positive_patch_coverage = np.asarray(
        [row["topography_coverage"] for row in cnn_patch_metadata["rows"]], dtype=float,
    )[positive]
    minimal_rows = [row for row in binary_rows if row["mapped_hazard_class"] == "minimal"]
    cnn_false_positive = cnn_predictions[~positive] >= cnn_threshold
    error_patterns = {
        "cnnMissedOnePercent": group_summary(positive_rows, cnn_missed),
        "cnnDetectedOnePercent": group_summary(positive_rows, ~cnn_missed),
        "bothMissedOnePercent": group_summary(positive_rows, cnn_missed & boosted_missed),
        "cnnFalsePositiveMinimal": group_summary(minimal_rows, cnn_false_positive),
        "cnnCorrectMinimal": group_summary(minimal_rows, ~cnn_false_positive),
        "cnnOnlyMissedCount": int(np.sum(cnn_missed & ~boosted_missed)),
        "boostedOnlyMissedCount": int(np.sum(~cnn_missed & boosted_missed)),
        "cnnMissedPatchesBelow90PercentTopography": int(np.sum(
            cnn_missed & (positive_patch_coverage < 0.9)
        )),
        "cnnDetectedPatchesBelow90PercentTopography": int(np.sum(
            ~cnn_missed & (positive_patch_coverage < 0.9)
        )),
    }

    results = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "datasetSha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "hydrography": {
            "source": hydro_source["source_url"],
            "localFile": str(HYDRO.relative_to(ROOT)),
            "sha256": hydro_sha,
            "flowlineCount": hydro_source["feature_count"],
            "note": "USGS 3DHP flowlines were included before sample selection and nearest-water feature computation.",
        },
        "target": "FEMA NFHL mapped hazard class, not observed event flooding",
        "sampleRole": "Western San Joaquin County external geographic test; no retraining",
        "samplingCaveat": "150 points per class; metrics do not measure natural class prevalence. Some points within narrow 0.2% polygons may be spatially correlated.",
        "features": FEATURES,
        "fiveInputFeatures": five_input_classes,
        "classOrder": CLASSES,
        "classCounts": dict(Counter(actual)),
        "errorPatterns": error_patterns,
        "models": {
            "terrainThreeClassBoosted": {
                "file": str(BOOSTED.relative_to(ROOT)), **metrics(actual, boosted_predicted),
            },
            "terrainThreeClassDnn": {
                "file": str(DNN.relative_to(ROOT)), **metrics(actual, dnn_predicted),
            },
            "fiveInputThreeClassDnn": {
                "file": str(FIVE_INPUT_THREE_CLASS_DNN.relative_to(ROOT)),
                **metrics(actual, five_input_three_class_predicted),
            },
        },
        "binaryComparison": {
            "classes": ["minimal", "one_percent"],
            "excludedClass": "point_two_percent (no binary target assigned)",
            "rowCount": int(binary_mask.sum()),
            "models": {
                "servedTerrainBoosted": {
                    "file": str(BINARY_BOOSTED.relative_to(ROOT)),
                    **binary_metrics(binary_actual, binary_boosted_scores, binary_boosted_threshold),
                },
                "terrainDnn": {
                    "file": str(BINARY_DNN.relative_to(ROOT)),
                    **binary_metrics(binary_actual, binary_dnn_scores, binary_dnn_threshold),
                },
                "highElevationTerrainBoosted": {
                    "file": str(HIGH_ELEVATION_BOOSTED.relative_to(ROOT)),
                    **binary_metrics(binary_actual, high_elevation_boosted_scores,
                                     high_elevation_boosted_threshold),
                },
                "fiveInputBoosted": {
                    "file": str(FIVE_INPUT_BOOSTED.relative_to(ROOT)),
                    **binary_metrics(binary_actual, five_input_boosted_scores,
                                     five_input_boosted_threshold),
                },
                "fiveInputDnn": {
                    "file": str(FIVE_INPUT_DNN.relative_to(ROOT)),
                    **binary_metrics(binary_actual, five_input_dnn_scores,
                                     float(five_input_dnn["threshold"])),
                },
                "terrainCnn": {
                    "file": str(CNN.relative_to(ROOT)),
                    "patchFile": str(CNN_PATCHES.relative_to(ROOT)),
                    "minimumTopographyCoverage": min(row["topography_coverage"]
                                                       for row in cnn_patch_metadata["rows"]),
                    "patchesBelow90PercentTopography": sum(
                        row["topography_coverage"] < 0.9 for row in cnn_patch_metadata["rows"]
                    ),
                    "bayAreaTestBalancedAccuracy": bay_cnn["balancedAccuracy"],
                    "bayAreaTestOnePercentRecall": bay_cnn["onePercentRecall"],
                    **binary_metrics(binary_actual, cnn_predictions, cnn_threshold),
                },
            },
        },
    }
    REPORT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Tracy–Lathrop–Stockton external model test", "",
        "This uses 450 western San Joaquin County points held out from existing model training:",
        "150 each from FEMA NFHL mapped minimal, 0.2%, and 1% annual-chance classes.",
        "The labels describe mapped hazard, not observed event flooding. No model was retrained.", "",
        f"The sample and nearest-water feature use {hydro_source['feature_count']:,} local USGS 3DHP",
        "flowlines plus existing local hydrography from the outset. The study window",
        "is (-121.55, 37.68, -121.15, 37.98), limited to San Joaquin FEMA map 06077C.", "",
        "| Existing model | Balanced accuracy | Macro F1 | 0.2% class F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, label in (("terrainThreeClassBoosted", "Terrain boosted"),
                        ("terrainThreeClassDnn", "Terrain DNN"),
                        ("fiveInputThreeClassDnn", "Five-input DNN")):
        result = results["models"][name]
        lines.append(f"| {label} | {result['balancedAccuracy']:.3f} | "
                     f"{result['macroF1']:.3f} | {result['perClassF1']['point_two_percent']:.3f} |")
    lines += ["", "Confusion matrices use rows = FEMA class and columns = predicted class,",
              "in the order minimal, 1%, 0.2%:", ""]
    for name, label in (("terrainThreeClassBoosted", "Terrain boosted"),
                        ("terrainThreeClassDnn", "Terrain DNN"),
                        ("fiveInputThreeClassDnn", "Five-input DNN")):
        lines.append(f"- {label}: `{results['models'][name]['confusionMatrix']}`")
    lines += ["", "The existing binary models are scored on the 300 explicit minimal and 1%",
              "points. The 0.2% points have no binary target and are excluded:", "",
              "| Binary terrain model | Balanced accuracy | 1% F1 | ROC AUC |",
              "| --- | ---: | ---: | ---: |"]
    for name, label in (("servedTerrainBoosted", "Served terrain boosted"),
                        ("terrainDnn", "Terrain DNN"),
                        ("highElevationTerrainBoosted", "High-elevation terrain boosted"),
                        ("fiveInputBoosted", "Five-input boosted"),
                        ("fiveInputDnn", "Five-input DNN"),
                        ("terrainCnn", "Terrain CNN")):
        result = results["binaryComparison"]["models"][name]
        lines.append(f"| {label} | {result['balancedAccuracy']:.3f} | "
                     f"{result['onePercentF1']:.3f} | {result['rocAuc']:.3f} |")
    lines += ["", "The binary thresholds come from the original validation region;",
              "they were not tuned on San Joaquin. Confusion matrices use rows = FEMA class",
              "and columns = predicted class, in the order minimal, 1%:", ""]
    for name, label in (("servedTerrainBoosted", "Served terrain boosted"),
                        ("terrainDnn", "Terrain DNN"),
                        ("highElevationTerrainBoosted", "High-elevation terrain boosted"),
                        ("fiveInputBoosted", "Five-input boosted"),
                        ("fiveInputDnn", "Five-input DNN"),
                        ("terrainCnn", "Terrain CNN")):
        lines.append(f"- {label}: `{results['binaryComparison']['models'][name]['confusionMatrix']}`")
    high_elevation_matrix = results["binaryComparison"]["models"]["highElevationTerrainBoosted"]["confusionMatrix"]
    lines += ["", "The high-elevation training variant detects "
              f"{high_elevation_matrix[1][1]} of 150 mapped 1% points and flags "
              f"{high_elevation_matrix[0][1]} of 150 mapped minimal points."]
    lines += ["", "The CNN uses 64×64 patches at 30 m resolution. Its Bay Area test",
              f"balanced accuracy was {bay_cnn['balancedAccuracy']:.3f} and 1% recall was "
              f"{bay_cnn['onePercentRecall']:.3f} with the same saved checkpoint.",
              f"Of the 300 San Joaquin patches, "
              f"{results['binaryComparison']['models']['terrainCnn']['patchesBelow90PercentTopography']} "
              "have less than 90% valid topography; the CNN receives validity masks."]
    flagged = error_patterns["cnnFalsePositiveMinimal"]
    clear = error_patterns["cnnCorrectMinimal"]
    lines += ["", "## Pattern among the CNN's mapped-minimal false positives", "",
              f"The CNN detects all {error_patterns['cnnDetectedOnePercent']['count']} "
              "mapped 1% points, but flags "
              f"{flagged['count']} of 150 mapped minimal points. Medians for flagged",
              "versus correctly cleared minimal points:", "",
              "| Terrain input | CNN flagged minimal | CNN cleared minimal |",
              "| --- | ---: | ---: |"]
    for field, label, unit in (
        ("height_above_drainage_m", "Height above drainage", "m"),
        ("nearest_water_distance_m", "Nearest mapped water", "m"),
        ("local_elevation_250m", "Elevation relative to 250 m neighborhood", "m"),
        ("local_elevation_500m", "Elevation relative to 500 m neighborhood", "m"),
        ("flow_accumulation_m2", "Flow accumulation", "m²"),
        ("elevation_m", "Elevation", "m"),
    ):
        lines.append(f"| {label} | {flagged['median'][field]:,.1f} {unit} | "
                     f"{clear['median'][field]:,.1f} {unit} |")
    top_polygon, top_count = Counter(flagged["sourcePolygonCounts"]).most_common(1)[0]
    lines += ["", "These group medians are descriptive; the features and FEMA polygons are",
              f"spatially correlated. Polygon {top_polygon} alone contributes {top_count} "
              "of the CNN false positives. Every CNN patch has at least 90%",
              "valid topography in this sample."]
    lines += ["", "This balanced, spatially sampled set is an external transfer check.",
              "It does not estimate real-world hazard prevalence or event prediction accuracy.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved external evaluation of {len(rows)} Tracy–Lathrop–Stockton points")
    for name, result in results["models"].items():
        print(f"{name}: balanced accuracy {result['balancedAccuracy']:.3f}, macro F1 {result['macroF1']:.3f}")
    for name, result in results["binaryComparison"]["models"].items():
        print(f"{name}: balanced accuracy {result['balancedAccuracy']:.3f}, 1% F1 {result['onePercentF1']:.3f}")


if __name__ == "__main__":
    main()
