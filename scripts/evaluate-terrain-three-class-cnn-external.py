#!/usr/bin/env python3
"""Compare the buffered three-class CNN and boosted model in Santa Cruz."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from terrain_features import vector
from terrain_three_class_metrics import CLASSES, evaluate


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/santa-cruz-north-fema-map-class-terrain-sample.csv"
DATASET_SOURCE = ROOT / "data/santa-cruz-north-fema-map-class-terrain-sample.source.json"
HYDRO_SOURCE = ROOT / "hydrography/santa_cruz_north_3dhp_flowlines.source.json"
HYDRO = ROOT / "hydrography/santa_cruz_north_3dhp_flowlines.geojson"
CACHE = ROOT / "data/cache/terrain-cnn/three-class"
REPORT = ROOT / "reports/terrain-three-class-cnn-external-evaluation.json"
REPORT_MD = ROOT / "reports/terrain-three-class-cnn-external-evaluation.md"


def main() -> None:
    with DATASET.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    dataset_sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    dataset_source = json.loads(DATASET_SOURCE.read_text(encoding="utf-8"))
    hydro_source = json.loads(HYDRO_SOURCE.read_text(encoding="utf-8"))
    hydro_sha = hashlib.sha256(HYDRO.read_bytes()).hexdigest()
    if dataset_source["supplemental_hydrography_sha256"] != hydro_sha or hydro_source["sha256"] != hydro_sha:
        raise RuntimeError("Santa Cruz hydrography does not match its source manifests")
    if len(rows) != 450 or {r["spatial_partition"] for r in rows} != {"external_test"}:
        raise RuntimeError("Expected the 450-point Santa Cruz external holdout")
    if {r["mapped_hazard_class"] for r in rows} != set(CLASSES):
        raise RuntimeError("The external holdout is missing a map class")
    with (ROOT / "data/dnn-three-class-training-table.csv").open(
        newline="", encoding="utf-8",
    ) as source:
        if "Santa Cruz County" in {r["county"] for r in csv.DictReader(source)}:
            raise RuntimeError("The external county appears in the training table")
    labels = np.asarray([CLASSES.index(r["mapped_hazard_class"]) for r in rows])
    x = np.asarray([vector({**r, "location_id": r["sample_id"]}, r)
                    for r in rows], dtype=np.float32)
    if not np.isfinite(x).all():
        raise RuntimeError("The external point features contain missing values")
    scorer = subprocess.run(
        [sys.executable, str(ROOT / "scripts/score-terrain-three-class-cnn-external.py")],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    cnn_probabilities = np.asarray(json.loads(scorer.stdout)["probabilities"])
    cnn = evaluate(labels, cnn_probabilities)
    boosted_model = joblib.load(CACHE / "baseline.joblib")
    boosted = evaluate(labels, boosted_model.predict_proba(x))
    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "datasetSha256": dataset_sha,
        "hydrographyCorrection": {
            "source": hydro_source["source_url"],
            "localFile": str(HYDRO.relative_to(ROOT)),
            "sha256": hydro_sha,
            "flowlineCount": hydro_source["feature_count"],
            "note": "USGS 3DHP flowlines supplement the local water network. All 450 sample IDs, coordinates, and labels stayed fixed; earlier Santa Cruz scores are superseded.",
        },
        "sampleRole": "Santa Cruz County external geographic test; no retraining or threshold changes",
        "samplingCaveat": "150 intentionally balanced points per class; class frequencies do not represent prevalence",
        "classes": CLASSES,
        "rows": len(rows),
        "cnn": cnn,
        "bufferedBoosted": boosted,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Three-class CNN external geographic evaluation", "",
        "The saved CNN and its buffered boosted comparator were scored without",
        "retraining on 450 northern Santa Cruz FEMA map-class points, 150 per class.",
        "This region is absent from the Bay Area training table.", "",
        f"This report uses the corrected water-distance inputs with {hydro_source['feature_count']:,}",
        "USGS 3DHP flowlines. The 450 point locations and labels are unchanged;",
        "the earlier three-class CNN Santa Cruz scores are superseded.", "",
        "| Metric | CNN | Buffered boosted |", "| --- | ---: | ---: |",
    ]
    for label, key in (("Accuracy", "accuracy"), ("Balanced accuracy", "balancedAccuracy"),
                       ("Macro F1", "macroF1"), ("Log loss", "logLoss")):
        lines.append(f"| {label} | {cnn[key]:.3f} | {boosted[key]:.3f} |")
    lines += ["", "| Class | CNN F1 | Boosted F1 |", "| --- | ---: | ---: |"]
    for name in CLASSES:
        lines.append(f"| {name} | {cnn['perClass'][name]['f1']:.3f} | "
                     f"{boosted['perClass'][name]['f1']:.3f} |")
    lines += ["", f"CNN confusion matrix: `{cnn['confusionMatrix']}`.",
              f"Boosted confusion matrix: `{boosted['confusionMatrix']}`.", "",
              "The sample is intentionally balanced and its points can be spatially",
              "correlated within narrow polygons. The labels are FEMA map categories,",
              "not observed floods or property flood probabilities.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Santa Cruz CNN macro F1 {cnn['macroF1']:.3f}, "
          f"boosted {boosted['macroF1']:.3f}; 0.2% F1 "
          f"{cnn['perClass']['point_two_percent']['f1']:.3f} vs "
          f"{boosted['perClass']['point_two_percent']['f1']:.3f}")


if __name__ == "__main__":
    main()
