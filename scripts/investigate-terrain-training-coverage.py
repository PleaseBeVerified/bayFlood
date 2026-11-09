"""Audit terrain coverage; optionally run paired, offline reweighting pilots.

Uses the existing CNN geographic buffers for every family. External regions
are diagnostic only and never used to compute weights or choose thresholds.
No production model artifacts are written.
"""

import argparse
import csv
import hashlib
import json
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from terrain_features import FEATURES, joined_rows, vector
from terrain_coverage import coverage_weights, terrain_stratum

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["minimal", "one_percent", "point_two_percent"]
TABLE = ROOT / "data/dnn-three-class-training-table.csv"
CALIBRATION = ROOT / "data/physical-hazard-calibration.csv"
CACHE = ROOT / "data/cache/terrain-cnn/three-class"
AUDIT = ROOT / "reports/terrain-training-coverage.json"
WEIGHTS = ROOT / "data/terrain-coverage-training-weights.csv"
PILOT = ROOT / "reports/terrain-coverage-weighting-pilot.json"
SEEDS = (42, 43, 44)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts_by_class(rows, values):
    result = {}
    for label in CLASSES:
        picks = [i for i, row in enumerate(rows) if row["mapped_hazard_class"] == label]
        a = values[picks]
        result[label] = {
            "count": len(picks),
            "flatLowHandCount": int(np.sum((a[:, 5] < 1) & (a[:, 9] < 3))),
            "lowFlatLowHandCount": int(np.sum((a[:, 0] < 20) & (a[:, 5] < 1) & (a[:, 9] < 3))),
            "elevationAtLeast50mCount": int(np.sum(a[:, 0] >= 50)),
            "higherRiverValleyCount": int(np.sum((a[:, 0] >= 50) & (a[:, 9] >= 3) & (a[:, 7] < -5))),
            "countyCounts": dict(Counter(rows[i]["county"] for i in picks)),
        }
    return result


def load_data():
    rows, mapping = joined_rows(TABLE, CALIBRATION, "mapped_hazard_class")
    metadata = json.loads((CACHE / "metadata.json").read_text())
    assert [r["location_id"] for r in rows] == [r["location_id"] for r in metadata["rows"]]
    x = np.asarray([mapping[r["location_id"]] for r in rows], dtype=float)
    y = np.asarray([CLASSES.index(r["mapped_hazard_class"]) for r in rows])
    xy = np.asarray([[r["x_m"], r["y_m"]] for r in metadata["rows"]])
    partitions = {k: np.asarray(v, dtype=int)
                  for k, v in json.loads((CACHE / "partitions.json").read_text()).items()}
    assert set(partitions) == {"training", "validation", "test"}
    assert not set(partitions["training"]) & set(np.concatenate((partitions["validation"], partitions["test"])))
    for name, indexes in partitions.items():
        assert all(rows[i]["spatial_partition"] == name for i in indexes)
    holdout = np.concatenate((partitions["validation"], partitions["test"]))
    separation = cKDTree(xy[holdout]).query(xy[partitions["training"]], p=np.inf)[0].min()
    assert separation > 1920
    external = {}
    for name in ("santa-cruz-north", "tracy-lathrop-stockton"):
        path = ROOT / f"data/{name}-fema-map-class-terrain-sample.csv"
        samples = list(csv.DictReader(path.open()))
        assert {r["spatial_partition"] for r in samples} == {"external_test"}
        assert not {r["county"] for r in samples} & {r["county"] for r in rows}
        external[name] = {
            "rows": samples,
            "x": np.asarray([vector({**r, "location_id": r["sample_id"]}, r) for r in samples]),
            "y": np.asarray([CLASSES.index(r["mapped_hazard_class"]) for r in samples]),
            "sha256": sha(path),
        }
    return rows, x, y, xy, partitions, external, float(separation)


def audit(rows, x, y, xy, partitions, external, separation):
    train = partitions["training"]
    w3 = coverage_weights(y[train], [rows[i]["county"] for i in train], x[train], xy[train])
    binary = y[train] != 2
    w2 = coverage_weights(y[train][binary], [rows[i]["county"] for i in train[binary]],
                          x[train][binary], xy[train][binary])
    binary_weights = dict(zip(train[binary], w2))
    with WEIGHTS.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["location_id", "county", "mapped_hazard_class", "spatial_partition",
                         "terrain_stratum", "spatial_block_2km", "coverage_weight_three_class",
                         "coverage_weight_binary"])
        for i, weight in zip(train, w3):
            writer.writerow([rows[i]["location_id"], rows[i]["county"], CLASSES[y[i]], "training",
                             terrain_stratum(x[i]), ":".join(map(str, np.floor(xy[i] / 2000).astype(int))),
                             f"{weight:.9f}", f"{binary_weights[i]:.9f}" if i in binary_weights else ""])
    training_counts = Counter((int(y[i]), terrain_stratum(x[i])) for i in train)
    support = {}
    for region, data in external.items():
        support[region] = {}
        for label in range(3):
            indexes = np.flatnonzero(data["y"] == label)
            n = [training_counts[(label, terrain_stratum(data["x"][i]))] for i in indexes]
            support[region][CLASSES[label]] = {
                "count": len(indexes), "noSameClassTerrainCellInTraining": sum(v == 0 for v in n),
                "fewerThan10SameClassTerrainCellRows": sum(v < 10 for v in n),
            }
    raw_training = np.asarray([i for i, r in enumerate(rows) if r["spatial_partition"] == "training"])
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceSha256": {str(p.relative_to(ROOT)): sha(p) for p in (TABLE, CALIBRATION, CACHE / "partitions.json")},
        "partitionCounts": {k: len(v) for k, v in partitions.items()},
        "minimumTrainingHoldoutChebyshevSeparationMeters": separation,
        "unbufferedTraining": counts_by_class([rows[i] for i in raw_training], x[raw_training]),
        "bufferedTraining": counts_by_class([rows[i] for i in train], x[train]),
        "external": {name: counts_by_class(d["rows"], d["x"]) for name, d in external.items()},
        "externalCellSupport": support,
        "strata": {"elevationEdgesMeters": [0, 5, 20, 50, 150], "handEdgesMeters": [1, 3, 10, 30],
                   "slopeEdgesDegrees": [1, 5, 15], "spatialBlockMeters": 2000},
        "weighting": {"meaning": "Mild inverse-square-root terrain, county, and within-terrain block density; class totals preserved",
                      "recipeSha256": sha(ROOT / "scripts/terrain_coverage.py"),
                      "finalBounds": [.25, 4], "trainingOnly": True,
                      "threeClassEffectiveSampleSize": float(w3.sum() ** 2 / np.square(w3).sum()),
                      "binaryEffectiveSampleSize": float(w2.sum() ** 2 / np.square(w2).sum()),
                      "emptyCellsCannotBeFixedByWeights": True},
        "provenanceGap": "Original calibration rows do not carry source polygon IDs or watershed IDs; a 2 km block is only a provisional clustering proxy.",
        "testStatus": "Santa Cruz and San Joaquin diagnostics motivated this design; they are development diagnostics, not untouched final tests.",
    }
    AUDIT.write_text(json.dumps(report, indent=2) + "\n")
    WEIGHTS.with_suffix(".source.json").write_text(json.dumps({
        "dataset_name": "Experimental terrain-coverage weights for existing buffered Bay training IDs",
        "dataset_role": "training-only development weights; no new examples or labels",
        "created_utc": report["generatedAt"], "row_count": len(train),
        "binary_row_count": int(binary.sum()), "partition": "training",
        "external_rows_included": 0, "class_order": CLASSES,
        "weights_sha256": sha(WEIGHTS), "recipe_sha256": sha(ROOT / "scripts/terrain_coverage.py"),
        "source_sha256": report["sourceSha256"],
        "limitations": "Only IDs in the actual buffered training subset may use these weights. Recompute from training-only data if splits change. Experimental; not selected for production.",
    }, indent=2) + "\n")
    lines = ["# Terrain training coverage audit", "",
             "Counts use the shared 1,920 m CNN geographic buffer. No external rows enter training.", "",
             "| Class | Training points | Flat, HAND <3 m | Elevation ≥50 m | Elevated river valley |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, group in report["bufferedTraining"].items():
        lines.append(f"| {label} | {group['count']} | {group['flatLowHandCount']} | "
                     f"{group['elevationAtLeast50mCount']} | {group['higherRiverValleyCount']} |")
    lines += ["", "Flat = slope <1°. Elevated river valley = elevation ≥50 m, HAND ≥3 m,",
              "and elevation relative to the 500 m neighborhood <−5 m. These are sampling strata.", "",
              "| External region | Class | No matching training cell | Fewer than 10 matching rows |",
              "| --- | --- | ---: | ---: |"]
    for region, groups in support.items():
        for label, group in groups.items():
            lines.append(f"| {region} | {label} | {group['noSameClassTerrainCellInTraining']}/{group['count']} | "
                         f"{group['fewerThan10SameClassTerrainCellRows']}/{group['count']} |")
    lines += ["", "Cells cross elevation, HAND, and slope bins listed in the adjacent JSON.", "",
              "Training-only weight file: `data/terrain-coverage-training-weights.csv`.",
              "Weights retain every training row, preserve class totals, and stay within 0.25–4.", "",
              report["provenanceGap"], "", report["testStatus"], "",
              "See [the acquisition and validation plan](../docs/terrain-generalization-training-plan.md)",
              "and [paired weighting pilot](terrain-coverage-weighting-pilot.md).", ""]
    AUDIT.with_suffix(".md").write_text("\n".join(lines))
    print("Saved coverage audit and training-only weights", flush=True)


def metrics(actual, probabilities, binary, threshold=None):
    predicted = (probabilities[:, 1] >= threshold).astype(int) if binary else probabilities.argmax(axis=1)
    result = {
        "balancedAccuracy": float(balanced_accuracy_score(actual, predicted)),
        "macroF1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "confusionMatrix": confusion_matrix(actual, predicted, labels=[0, 1] if binary else [0, 1, 2]).tolist(),
    }
    if binary:
        result["rocAuc"] = float(roc_auc_score(actual, probabilities[:, 1]))
    return result


def pilot(rows, x, y, xy, partitions, external):
    configurations = [
        ("binary_five_input_boosted", True, 5, "boosted"),
        ("binary_terrain_boosted", True, 10, "boosted"),
        ("binary_five_input_dnn", True, 5, "dnn"),
        ("binary_terrain_dnn", True, 10, "dnn"),
        ("three_class_five_input_dnn", False, 5, "dnn"),
        ("three_class_terrain_boosted", False, 10, "boosted"),
        ("three_class_terrain_dnn", False, 10, "dnn"),
    ]
    results = []
    for name, binary, nfeatures, family in configurations:
        indexes = {k: v[y[v] != 2] if binary else v for k, v in partitions.items()}
        train, validation = indexes["training"], indexes["validation"]
        weights = coverage_weights(y[train], [rows[i]["county"] for i in train], x[train], xy[train])
        base = np.ones(len(train))
        if not binary:
            class_counts = Counter(y[train])
            base = np.asarray([len(train) / 3 / class_counts[v] for v in y[train]])
        imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
        train_x = scaler.fit_transform(imputer.fit_transform(x[train, :nfeatures])) if family == "dnn" else x[train, :nfeatures]
        def transform(values):
            values = values[:, :nfeatures]
            return scaler.transform(imputer.transform(values)) if family == "dnn" else values
        for seed in SEEDS:
            for variant, multiplier in (("control", np.ones(len(train))), ("coverage", weights)):
                model = XGBClassifier(
                    n_estimators=350, max_depth=3, learning_rate=.03, subsample=.85,
                    colsample_bytree=.9, reg_lambda=10, n_jobs=4, random_state=seed,
                    objective="binary:logistic" if binary else "multi:softprob",
                ) if family == "boosted" else MLPClassifier(
                    hidden_layer_sizes=(64, 32), alpha=.01, batch_size=128,
                    max_iter=500, early_stopping=False, n_iter_no_change=20, random_state=seed,
                )
                started = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(train_x, y[train], sample_weight=base * multiplier)
                validation_probabilities = model.predict_proba(transform(x[validation]))
                threshold = None
                if binary:
                    positive_scores = np.sort(validation_probabilities[y[validation] == 1, 1])
                    threshold = float(positive_scores[len(positive_scores) - int(np.ceil(.95 * len(positive_scores)))])
                evaluation = {
                    "bay_validation": metrics(y[validation], validation_probabilities, binary, threshold),
                    "bay_test": metrics(y[indexes["test"]], model.predict_proba(transform(x[indexes["test"]])), binary, threshold),
                }
                for region, data in external.items():
                    keep = data["y"] != 2 if binary else np.ones(len(data["y"]), dtype=bool)
                    evaluation[region] = metrics(data["y"][keep], model.predict_proba(transform(data["x"][keep])), binary, threshold)
                results.append({
                    "model": name, "seed": seed, "variant": variant,
                    "trainingRows": len(train), "validationThreshold": threshold,
                    "trainingSeconds": time.monotonic() - started,
                    "fitWarnings": [str(w.message) for w in caught],
                    "evaluation": evaluation,
                })
                print(f"{name} seed={seed} {variant}: " + ", ".join(
                    f"{region} BA={m['balancedAccuracy']:.3f}" for region, m in evaluation.items()
                    if region != "bay_validation"), flush=True)
                PILOT.write_text(json.dumps({"complete": False, "runs": results}, indent=2) + "\n")
    summarize_pilot(results)


def summarize_pilot(results):
    assert len(results) == 42
    aggregate = {}
    names = list(dict.fromkeys(r["model"] for r in results))
    for name in names:
        aggregate[name] = {}
        for region in ("bay_test", "santa-cruz-north", "tracy-lathrop-stockton"):
            aggregate[name][region] = {}
            for variant in ("control", "coverage"):
                picks = [r["evaluation"][region] for r in results if r["model"] == name and r["variant"] == variant]
                aggregate[name][region][variant] = {
                    metric: float(np.mean([r[metric] for r in picks]))
                    for metric in ("balancedAccuracy", "macroF1")
                }
                matrices = np.asarray([r["confusionMatrix"] for r in picks], dtype=float)
                recall = np.diagonal(matrices, axis1=1, axis2=2) / matrices.sum(axis=2)
                aggregate[name][region][variant]["perClassRecall"] = {
                    label: float(value) for label, value in zip(CLASSES, recall.mean(axis=0))
                }
                aggregate[name][region][variant]["meanConfusionMatrix"] = matrices.mean(axis=0).tolist()
    report = {
        "complete": True, "generatedAt": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS, "pairedRuns": len(results), "runs": results, "aggregate": aggregate,
        "protocol": "Prespecified mild weighting; same rows, features, architecture, seed and preprocessing within pairs. Binary thresholds target 95% Bay validation recall. Three-class control and coverage both use class-balanced loss. DNN training uses a 500-iteration maximum and training-loss convergence; no candidate chosen on external scores.",
        "runsWithFitWarnings": sum(bool(r["fitWarnings"]) for r in results),
        "limitations": "Pilot configurations are not exact retrains of every saved checkpoint. CNNs were not retrained. Three seeds measure initialization variation, not independent regional uncertainty. Existing external results motivated the design; a fresh geographic test is required. No production model replaced.",
        "sourceSha256": json.loads(AUDIT.read_text())["sourceSha256"],
        "weightRecipeSha256": sha(ROOT / "scripts/terrain_coverage.py"),
        "trainingWeightsSha256": sha(WEIGHTS),
    }
    PILOT.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Training coverage weighting pilot", "", report["protocol"], "",
             "Each cell is mean balanced accuracy across three paired seeds: control → coverage weighting.", "",
             "| Model | Bay test | Santa Cruz | Tracy–Lathrop–Stockton |", "| --- | ---: | ---: | ---: |"]
    for name, regions in aggregate.items():
        cells = [f"{regions[r]['control']['balancedAccuracy']:.3f} → {regions[r]['coverage']['balancedAccuracy']:.3f}"
                 for r in ("bay_test", "santa-cruz-north", "tracy-lathrop-stockton")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Three-class 0.2% recall must also be inspected; higher balanced accuracy",
              "can coexist with abandoning that class:", "",
              "| Model | Santa Cruz 0.2% recall | San Joaquin 0.2% recall |",
              "| --- | ---: | ---: |"]
    for name, regions in aggregate.items():
        if not name.startswith("three_class"):
            continue
        cells = [f"{regions[r]['control']['perClassRecall']['point_two_percent']:.3f} → "
                 f"{regions[r]['coverage']['perClassRecall']['point_two_percent']:.3f}"
                 for r in ("santa-cruz-north", "tracy-lathrop-stockton")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"Runs with fit warnings: {report['runsWithFitWarnings']}/{len(results)}. "
              "Inspect the JSON for convergence warnings before treating DNN differences as stable.", "",
              report["limitations"], ""]
    PILOT.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--summarize-pilot", action="store_true")
    arguments = parser.parse_args()
    if arguments.summarize_pilot:
        summarize_pilot(json.loads(PILOT.read_text())["runs"])
        raise SystemExit(0)
    data = load_data()
    audit(*data)
    if arguments.pilot:
        pilot(*data[:-1])
