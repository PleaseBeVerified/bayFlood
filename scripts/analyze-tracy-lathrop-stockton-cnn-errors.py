"""Audit mapped-minimal false positives from the saved binary terrain CNN.

Run with the GDAL-enabled system Python. Model inference runs separately in
the trusted model environment. This analysis does not fit or tune a model.
"""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from osgeo import ogr, osr

from terrain_features import FEATURES, joined_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tracy-lathrop-stockton-fema-map-class-terrain-sample.csv"
CACHE = ROOT / "data/cache/terrain-cnn"
PREFIX = ROOT / "reports/tracy-lathrop-stockton-binary-cnn-false-positives"
FIELDS = [
    "elevation_m", "height_above_drainage_m", "slope_degrees",
    "nearest_water_distance_m", "current_imperviousness_pct",
    "current_100yr_24h_rainfall_mm", "local_elevation_250m",
    "local_elevation_500m", "flow_accumulation_m2",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows, fields):
    return {field: dict(zip(("q25", "median", "q75"),
                           np.quantile([float(r[field]) for r in rows], [.25, .5, .75]).tolist()))
            for field in fields} if rows else {}


def reference(code):
    result = osr.SpatialReference()
    result.ImportFromEPSG(code)
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def main():
    ogr.UseExceptions()
    osr.UseExceptions()
    rows = [r for r in csv.DictReader(DATA.open())
            if r["mapped_hazard_class"] in ("minimal", "one_percent")]
    metadata = json.loads((CACHE / "tracy-lathrop-stockton-external-patches.json").read_text())
    assert metadata["datasetSha256"] == digest(DATA)
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in metadata["rows"]]
    scored = subprocess.run([
        str(ROOT / ".venv-model/bin/python"), str(ROOT / "scripts/score-santa-cruz-north-cnn.py"),
        "--region", "tracy-lathrop-stockton",
    ], cwd=ROOT, capture_output=True, text=True, check=True)
    inference = json.loads(scored.stdout)
    threshold = inference["threshold"]
    assert len(inference["scores"]) == len(rows)
    patches = np.load(CACHE / "tracy-lathrop-stockton-external-patches-float16.npy", mmap_mode="r")
    for i, (row, score) in enumerate(zip(rows, inference["scores"])):
        row["cnn_score"] = score
        row["cnn_flagged"] = score >= threshold
        row["outcome"] = ("false_positive" if row["cnn_flagged"] else "true_negative") \
            if row["mapped_hazard_class"] == "minimal" else "true_positive" if row["cnn_flagged"] else "false_negative"
        patch = np.asarray(patches[i], dtype=float)
        valid = patch[7] > .5
        row["patch_median_slope_degrees"] = float(np.median(patch[0][valid]))
        row["patch_median_height_above_drainage_m"] = float(np.median(patch[4][valid]))
        row["patch_dem_iqr_m"] = float(np.diff(np.quantile(patch[5][valid], [.25, .75]))[0])
        row["topography_coverage"] = metadata["rows"][i]["topography_coverage"]
        row["imperviousness_coverage"] = metadata["rows"][i]["imperviousness_coverage"]

    minimal = [r for r in rows if r["mapped_hazard_class"] == "minimal"]
    fp = [r for r in minimal if r["cnn_flagged"]]
    tn = [r for r in minimal if not r["cnn_flagged"]]
    positive = [r for r in rows if r["mapped_hazard_class"] == "one_percent"]
    existing = json.loads((ROOT / "reports/tracy-lathrop-stockton-fema-terrain-external-evaluation.json").read_text())
    assert existing["datasetSha256"] == digest(DATA)
    assert [[len(tn), len(fp)], [sum(not r["cnn_flagged"] for r in positive),
                                sum(r["cnn_flagged"] for r in positive)]] == existing["binaryComparison"]["models"]["terrainCnn"]["confusionMatrix"]

    # Audit distance to the actual containing FEMA polygon boundary at stored coordinates.
    transform = osr.CoordinateTransformation(reference(4326), reference(3310))
    source_path = ROOT / "FEMAdata/tracy_lathrop_stockton_nfhl_flood_zones.geojson"
    source_manifest = json.loads(DATA.with_suffix(".source.json").read_text())
    assert source_manifest["source_sha256"] == digest(source_path)
    source = json.loads(source_path.read_text())
    polygons = {}
    properties = {}
    for feature in source["features"]:
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature["geometry"]))
        geometry.Transform(transform)
        object_id = str(feature["properties"]["OBJECTID"])
        polygons[object_id] = geometry
        properties[object_id] = feature["properties"]
    for row in minimal:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(float(row["lng"]), float(row["lat"]))
        point.Transform(transform)
        polygon = polygons[row["source_object_id"]]
        assert polygon.Contains(point), row["sample_id"]
        label_properties = properties[row["source_object_id"]]
        assert label_properties["FLD_ZONE"] == "X"
        assert label_properties["ZONE_SUBTY"] == "AREA OF MINIMAL FLOOD HAZARD"
        row["fema_polygon_boundary_distance_m"] = polygon.Boundary().Distance(point)

    patch_fields = ["patch_median_slope_degrees", "patch_median_height_above_drainage_m", "patch_dem_iqr_m"]
    diagnostic_fields = FIELDS + patch_fields + ["cnn_score", "fema_polygon_boundary_distance_m"]
    polygon_stats = []
    for object_id, count in Counter(r["source_object_id"] for r in minimal).most_common():
        group = [r for r in minimal if r["source_object_id"] == object_id]
        flagged = [r for r in group if r["cnn_flagged"]]
        clear = [r for r in group if not r["cnn_flagged"]]
        polygon_stats.append({
            "sourceObjectId": object_id, "minimalCount": count,
            "falsePositives": len(flagged), "falsePositiveRate": len(flagged) / count,
            "falsePositiveTerrain": summary(flagged, FIELDS),
            "correctMinimalTerrain": summary(clear, FIELDS),
        })
    bins = {}
    for field, breaks in (("elevation_m", [-float("inf"), 10, 20, 40, float("inf")]),
                          ("height_above_drainage_m", [-float("inf"), 1, 3, 10, float("inf")])):
        bins[field] = []
        for low, high in zip(breaks, breaks[1:]):
            group = [r for r in minimal if low <= float(r[field]) < high]
            bins[field].append({"range": f"[{low}, {high})", "minimalCount": len(group),
                                "falsePositives": sum(r["cnn_flagged"] for r in group)})
    thresholds = [{"threshold": t,
                   "minimalFalsePositives": sum(r["cnn_score"] >= t for r in minimal),
                   "onePercentDetected": sum(r["cnn_score"] >= t for r in positive)}
                  for t in (threshold, .1, .2, .5, .8)]

    bay_rows, bay_x = joined_rows(ROOT / "data/dnn-training-table.csv",
                                  ROOT / "data/physical-hazard-calibration.csv")
    bay_test = [r for r in bay_rows if r["spatial_partition"] == "test"]
    bay_comparison = {}
    for label in ("0", "1"):
        group = [dict(zip(FEATURES, bay_x[r["location_id"]]))
                 for r in bay_test if r["flood_risk_label"] == label]
        bay_comparison["minimal" if label == "0" else "one_percent"] = {
            "count": len(group), "terrain": summary(group, [f for f in FIELDS if f in group[0]]),
        }

    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "datasetSha256": digest(DATA), "modelSha256": digest(ROOT / "models/flood-risk-terrain-cnn.pt"),
        "sourceFemaSha256": digest(source_path), "threshold": threshold,
        "minimalCount": len(minimal), "falsePositiveCount": len(fp),
        "falsePositiveRate": len(fp) / len(minimal),
        "falsePositiveSummary": summary(fp, diagnostic_fields),
        "correctMinimalSummary": summary(tn, diagnostic_fields),
        "mappedOnePercentSummary": summary(positive, FIELDS + patch_fields + ["cnn_score"]),
        "polygonGroups": polygon_stats, "terrainBins": bins,
        "thresholdSensitivityDiagnosticOnly": thresholds,
        "bayAreaTestInputComparison": bay_comparison,
        "qualityChecks": {
            "allMinimalPointsInsideRecordedSourcePolygon": True,
            "allMinimalLabelsExplicitFemaUnshadedZoneX": True,
            "minimumTopographyCoverage": min(r["topography_coverage"] for r in rows),
            "minimumImperviousnessCoverage": min(r["imperviousness_coverage"] for r in rows),
            "falsePositivesWithin30mOfFemaBoundary": sum(r["fema_polygon_boundary_distance_m"] < 30 for r in fp),
            "falsePositivesWithin100mOfFemaBoundary": sum(r["fema_polygon_boundary_distance_m"] < 100 for r in fp),
            "falsePositivesMoreThan1400mInsideFemaPolygon": sum(r["fema_polygon_boundary_distance_m"] > 1400 for r in fp),
        },
        "limitations": "Descriptive spatially correlated sample; associations do not isolate causal model drivers. Scores measure mapped-class agreement, not event probabilities. Threshold alternatives use test labels for diagnosis only and must not be selected for deployment from this analysis.",
    }
    PREFIX.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = list(minimal[0])
    with PREFIX.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(minimal)
    lines = ["# Binary CNN false positives: Tracy–Lathrop–Stockton", "",
             f"The saved CNN flags {len(fp)}/{len(minimal)} FEMA mapped minimal points "
             f"({len(fp)/len(minimal):.1%}) at its original Bay Area validation threshold ({threshold:.5f}).",
             "All 150 mapped 1% points are detected. No model or threshold was changed.", "",
             "## Findings", "",
             "The main descriptive pattern is low, flat terrain with low height above drainage,",
             "both at the sampled point and throughout its roughly 1.92 km wide image patch.",
             "Water proximity and flow accumulation overlap substantially between flagged and cleared",
             "minimal points, so those inputs alone give a weaker explanation.", "",
             "Two polygons contribute 92 of 104 false positives, but also contain 135 of 150",
             "minimal samples. The terrain pattern persists within polygon 28049960:",
             "flagged versus cleared median elevation is 13.1 versus 38.7 m, and height",
             "above drainage is 2.5 versus 8.7 m. Spatial correlation still limits inference.", "",
             "Complete raster coverage and the boundary audit provide no evidence that these",
             "errors are mainly missing pixels or tiny coordinate shifts across FEMA boundaries.", "",
             "The recall-oriented threshold amplifies the false positives. Raising the score",
             "threshold to 0.5 would still leave 42 false positives and reduce detection of",
             "mapped 1% points from 150 to 107. Threshold sensitivity does not justify",
             "selecting a new operating point on this external test set.", "",
             "## Terrain pattern", "", "Medians, with Q25–Q75 in parentheses:", "",
             "| Input | Flagged minimal | Correctly cleared minimal |", "| --- | ---: | ---: |"]
    for field in diagnostic_fields:
        left, right = report["falsePositiveSummary"][field], report["correctMinimalSummary"][field]
        lines.append(f"| {field} | {left['median']:.3f} ({left['q25']:.3f}–{left['q75']:.3f}) | "
                     f"{right['median']:.3f} ({right['q25']:.3f}–{right['q75']:.3f}) |")
    lines += ["", "## Spatial concentration", "", "| FEMA polygon | Minimal points | False positives | Rate |",
              "| --- | ---: | ---: | ---: |"]
    for p in polygon_stats:
        lines.append(f"| {p['sourceObjectId']} | {p['minimalCount']} | {p['falsePositives']} | {p['falsePositiveRate']:.1%} |")
    lines += ["", "Concentration must be interpreted against the number of sampled points per polygon.",
              "Polygon group medians and Bay Area test feature comparisons are in the adjacent JSON.", "",
              "## Bay Area input contrast", "",
              "Medians in the saved Bay Area test set versus the new region's false positives:", "",
              "| Feature | Bay minimal (812) | Bay mapped 1% (342) | New-region false positives (104) |",
              "| --- | ---: | ---: | ---: |"]
    for field in ("elevation_m", "height_above_drainage_m", "slope_degrees"):
        lines.append(f"| {field} | {bay_comparison['minimal']['terrain'][field]['median']:.3f} | "
                     f"{bay_comparison['one_percent']['terrain'][field]['median']:.3f} | "
                     f"{report['falsePositiveSummary'][field]['median']:.3f} |")
    lines += ["", "This is consistent with geographic transfer failure: terrain resembling the Bay",
              "lowland positive class can still have a minimal FEMA label in San Joaquin County.",
              "This association does not establish which CNN inputs cause an individual prediction.", "",
              "## Terrain strata", "", "| Feature | Range | False positives / minimal points |",
              "| --- | --- | ---: |"]
    for field, entries in bins.items():
        for entry in entries:
            lines.append(f"| {field} | {entry['range']} | {entry['falsePositives']}/{entry['minimalCount']} |")
    lines += ["", "## Score and threshold diagnostic", "",
              "The original threshold was selected for at least 95% recall on Bay Area validation.",
              "The threshold alternatives below are descriptive; they were not applied to the model.", "",
              "| Score threshold | Minimal false positives / 150 | Mapped 1% detected / 150 |",
              "| --- | ---: | ---: |"]
    for entry in thresholds:
        lines.append(f"| {entry['threshold']:.5f} | {entry['minimalFalsePositives']} | {entry['onePercentDetected']} |")
    lines += ["", "## Data checks and limits", "", "```json",
              json.dumps(report["qualityChecks"], indent=2), "```", "",
              report["limitations"], "",
              "Per-point audit: adjacent CSV. Reproduce with "
              "`python3 scripts/analyze-tracy-lathrop-stockton-cnn-errors.py`.", ""]
    PREFIX.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"falsePositives": len(fp), "thresholds": thresholds,
                      "quality": report["qualityChecks"]}, indent=2))


if __name__ == "__main__":
    main()
