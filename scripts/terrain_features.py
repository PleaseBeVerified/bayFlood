"""Shared ten-feature input contract for the terrain FEMA classifiers."""

from __future__ import annotations

import csv
import math
from pathlib import Path


FEATURES = [
    "elevation_m",
    "current_imperviousness_pct",
    "current_100yr_24h_rainfall_mm",
    "nearest_water_available",
    "log_nearest_water_distance_m",
    "slope_degrees",
    "local_elevation_250m",
    "local_elevation_500m",
    "log_flow_accumulation_m2",
    "height_above_drainage_m",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def vector(row: dict[str, str], terrain: dict[str, str]) -> list[float]:
    distance = row["nearest_water_distance_m"]
    accumulation = float(terrain["flow_accumulation_m2"])
    if accumulation < 0:
        raise ValueError(f"Negative flow accumulation: {row['location_id']}")
    return [
        float(row["elevation_m"]),
        float(row["current_imperviousness_pct"]),
        float(row["current_100yr_24h_rainfall_mm"]),
        float(row["nearest_water_available"]),
        math.log1p(max(0.0, float(distance))) if distance else math.nan,
        float(terrain["slope_degrees"]),
        float(terrain["local_elevation_250m"]),
        float(terrain["local_elevation_500m"]),
        math.log1p(accumulation),
        float(terrain["height_above_drainage_m"]),
    ]


def joined_rows(
    training_table: Path,
    calibration: Path,
    target_column: str = "flood_risk_label",
) -> tuple[list[dict[str, str]], dict[str, list[float]]]:
    calibration_rows = read_csv(calibration)
    table_rows = read_csv(training_table)
    terrain_by_id = {row["sample_id"]: row for row in calibration_rows}
    if len(terrain_by_id) != len(calibration_rows):
        raise RuntimeError("Duplicate calibration location IDs")
    if len({row["location_id"] for row in table_rows}) != len(table_rows):
        raise RuntimeError("Duplicate training-table location IDs")
    if any(row["location_id"] not in terrain_by_id for row in table_rows):
        raise RuntimeError("Training rows missing terrain features")
    if target_column not in {"flood_risk_label", "mapped_hazard_class"}:
        raise ValueError(f"Unsupported target column: {target_column}")
    if any(terrain_by_id[row["location_id"]]["mapped_hazard_class"] !=
           (row["mapped_hazard_class"] if target_column == "mapped_hazard_class"
            else "one_percent" if row["flood_risk_label"] == "1" else "minimal")
           for row in table_rows):
        raise RuntimeError("Training labels disagree with calibration labels")
    x_by_id = {row["location_id"]: vector(row, terrain_by_id[row["location_id"]])
               for row in table_rows}
    return table_rows, x_by_id
