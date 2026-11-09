#!/usr/bin/env python3
"""Score one already-extracted raster patch from a JSON request on stdin."""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from terrain_cnn_model import TerrainCNN


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/flood-risk-terrain-cnn.pt"
FEATURES = [
    "elevation_m", "current_imperviousness_pct", "current_100yr_24h_rainfall_mm",
    "nearest_water_available", "log_nearest_water_distance_m", "slope_degrees",
    "local_elevation_250m", "local_elevation_500m", "log_flow_accumulation_m2",
    "height_above_drainage_m",
]
EXPECTED_CHANNELS = [
    "slope_degrees", "local_elevation_250m", "local_elevation_500m",
    "log_flow_accumulation_m2", "height_above_drainage_m",
    "dem_relative_to_point_m", "imperviousness_pct",
    "topography_valid", "imperviousness_valid",
]


def main() -> None:
    torch.set_num_threads(2)
    request = json.load(sys.stdin)
    encoded = request.get("patchBase64")
    raw_features = request.get("features")
    if not isinstance(encoded, str) or not isinstance(raw_features, list) or len(raw_features) != len(FEATURES):
        raise ValueError("Expected one patch and ten point features")
    binary = base64.b64decode(encoded, validate=True)
    if len(binary) != 9 * 64 * 64 * 4:
        raise ValueError("Unexpected patch shape")
    patch = np.frombuffer(binary, dtype="<f4").reshape(9, 64, 64).copy()
    if not np.isfinite(patch).all() or not np.isin(patch[7:], [0, 1]).all():
        raise ValueError("Invalid patch values or masks")
    if patch[7, 32, 32] != 1:
        raise ValueError("Missing topography at selected location")
    features = np.asarray([math.nan if value is None else float(value)
                           for value in raw_features], dtype=np.float32)
    if not np.isfinite(features[~np.isnan(features)]).all():
        raise ValueError("Invalid point features")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["channels"] != EXPECTED_CHANNELS:
        raise RuntimeError("CNN checkpoint channel order does not match the raster patch")
    for channel in range(7):
        valid = patch[7 if channel < 6 else 8] > 0.5
        patch[channel, valid] = np.clip(
            (patch[channel, valid] - checkpoint["pixelMean"][channel]) /
            checkpoint["pixelScale"][channel], -6, 6,
        )
    median = np.asarray(checkpoint["tabularMedian"], dtype=np.float32)
    mean = np.asarray(checkpoint["tabularMean"], dtype=np.float32)
    scale = np.asarray(checkpoint["tabularScale"], dtype=np.float32)
    normalized = (np.where(np.isnan(features), median, features) - mean) / scale
    model = TerrainCNN()
    model.load_state_dict(checkpoint["stateDict"])
    model.eval()
    with torch.no_grad():
        probability = float(torch.sigmoid(model(
            torch.from_numpy(patch[None]), torch.from_numpy(normalized[None]),
        ))[0])
    threshold = float(checkpoint["threshold"])
    print(json.dumps({
        "onePercentPatternScore": probability,
        "classification": "higher-risk pattern" if probability >= threshold else "lower-risk pattern",
        "imputedFeatures": [name for name, value in zip(FEATURES, features, strict=True)
                            if math.isnan(float(value))],
        "topographyCoverage": float(patch[7].mean()),
        "imperviousnessCoverage": float(patch[8].mean()),
    }), flush=True)


if __name__ == "__main__":
    main()
