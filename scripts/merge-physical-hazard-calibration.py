#!/usr/bin/env python3
"""Audit and merge targeted physical-hazard calibration rows safely."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "physical-hazard-calibration.csv"
ADDITIONS_PATH = ROOT / "data" / "physical-hazard-point-two-extra.csv"
OUTPUT_PATH = ROOT / "data" / "physical-hazard-calibration-expanded.csv"
REPORT_PATH = ROOT / "reports" / "physical-hazard-point-two-merge.json"
MINIMUM_DISTANCE_METERS = 30.0
REQUIRED_NUMERIC = [
    "lat",
    "lng",
    "elevation_m",
    "rainfall_100yr_24h_in",
    "slope_degrees",
    "local_elevation_250m",
    "local_elevation_500m",
    "flow_accumulation_m2",
    "height_above_drainage_m",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path.relative_to(ROOT)} has no header")
        return reader.fieldnames, list(reader)


def fnv1a_32(value: str) -> int:
    hash_value = 2_166_136_261
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 16_777_619) & 0xFFFFFFFF
    return hash_value


def block_key(row: dict[str, str]) -> str:
    return (
        f"{math.floor(float(row['lat']) / 0.05)},"
        f"{math.floor(float(row['lng']) / 0.05)}"
    )


def partition(row: dict[str, str]) -> str:
    bucket = fnv1a_32(block_key(row)) % 20
    if bucket < 14:
        return "training"
    if bucket < 17:
        return "validation"
    return "test"


def coordinate(row: dict[str, str]) -> tuple[float, float]:
    return float(row["lat"]), float(row["lng"])


def distance_meters(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    latitude = math.radians((first[0] + second[0]) / 2)
    north = (first[0] - second[0]) * 110_540
    east = (first[1] - second[1]) * 111_320 * math.cos(latitude)
    return math.hypot(north, east)


def spatial_cell(point: tuple[float, float]) -> tuple[int, int]:
    # Roughly 25-meter cells; a 30-meter search only needs adjacent cells.
    return math.floor(point[0] / 0.000225), math.floor(point[1] / 0.000225)


def nearby_coordinates(
    point: tuple[float, float],
    index: dict[tuple[int, int], list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    row, column = spatial_cell(point)
    return [
        candidate
        for row_offset in range(-2, 3)
        for column_offset in range(-2, 3)
        for candidate in index.get((row + row_offset, column + column_offset), [])
    ]


def add_to_index(
    point: tuple[float, float],
    index: dict[tuple[int, int], list[tuple[float, float]]],
) -> None:
    index.setdefault(spatial_cell(point), []).append(point)


def validate_numeric(row: dict[str, str]) -> str | None:
    for field in REQUIRED_NUMERIC:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            return f"invalid_{field}"
        if not math.isfinite(value):
            return f"invalid_{field}"
    return None


def summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "classCounts": dict(Counter(row["mapped_hazard_class"] for row in rows)),
        "partitionCounts": dict(Counter(partition(row) for row in rows)),
        "countyCounts": dict(Counter(row["county"] for row in rows)),
        "geographicBlocks": len({block_key(row) for row in rows}),
    }


def main() -> None:
    headers, base_rows = read_csv(BASE_PATH)
    addition_headers, addition_rows = read_csv(ADDITIONS_PATH)
    if addition_headers != headers:
        raise RuntimeError("The additions CSV columns do not match the base table")

    base_ids = {row["sample_id"] for row in base_rows}
    base_exact = {coordinate(row) for row in base_rows}
    coordinate_index: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for point in base_exact:
        add_to_index(point, coordinate_index)

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    accepted_ids: set[str] = set()
    accepted_exact: set[tuple[float, float]] = set()
    for row in sorted(addition_rows, key=lambda item: item["sample_id"]):
        reason = validate_numeric(row)
        point: tuple[float, float] | None = None
        if reason is None:
            point = coordinate(row)
            if row["mapped_hazard_class"] != "point_two_percent":
                reason = "wrong_hazard_class"
            elif partition(row) == "test":
                reason = "untouched_test_block"
            elif row["sample_id"] in base_ids or row["sample_id"] in accepted_ids:
                reason = "duplicate_sample_id"
            elif point in base_exact or point in accepted_exact:
                reason = "exact_duplicate_coordinate"
            elif any(
                distance_meters(point, candidate) < MINIMUM_DISTANCE_METERS
                for candidate in nearby_coordinates(point, coordinate_index)
            ):
                reason = "within_30m_of_existing_or_retained_row"
        if reason is not None:
            rejected.append({"sampleId": row.get("sample_id", ""), "reason": reason})
            continue
        assert point is not None
        accepted.append(row)
        accepted_ids.add(row["sample_id"])
        accepted_exact.add(point)
        add_to_index(point, coordinate_index)

    expanded = [*base_rows, *accepted]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(expanded)

    base_test_signature = {
        (row["sample_id"], row["lat"], row["lng"])
        for row in base_rows
        if partition(row) == "test"
    }
    expanded_test_signature = {
        (row["sample_id"], row["lat"], row["lng"])
        for row in expanded
        if partition(row) == "test"
    }
    if base_test_signature != expanded_test_signature:
        raise RuntimeError("The merge changed the untouched test rows")

    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "base": str(BASE_PATH.relative_to(ROOT)),
        "additions": str(ADDITIONS_PATH.relative_to(ROOT)),
        "output": str(OUTPUT_PATH.relative_to(ROOT)),
        "minimumDistanceMeters": MINIMUM_DISTANCE_METERS,
        "baseSummary": summary(base_rows),
        "additionSummary": summary(addition_rows),
        "acceptedSummary": summary(accepted),
        "expandedSummary": summary(expanded),
        "rejectedCount": len(rejected),
        "rejectedReasons": dict(Counter(item["reason"] for item in rejected)),
        "rejectedRows": rejected,
        "untouchedTestRowsPreserved": True,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Merged calibration rows:",
        {
            "base": len(base_rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "expanded": len(expanded),
            "untouchedTestRows": len(base_test_signature),
        },
        flush=True,
    )
    print("Rejected reasons:", report["rejectedReasons"], flush=True)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}", flush=True)
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
