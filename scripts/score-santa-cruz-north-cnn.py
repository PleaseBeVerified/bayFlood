"""Score an external-region CNN patch cache in an isolated PyTorch process.

The parent comparison process loads XGBoost, which conflicts with PyTorch's
native runtime in this environment. Output is one JSON object on stdout.
"""

import csv
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from terrain_cnn_model import TerrainCNN
from terrain_features import vector


ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("santa-cruz-north", "tracy-lathrop-stockton")
CHECKPOINT = ROOT / "models/flood-risk-terrain-cnn.pt"
CHANNELS = [
    "slope_degrees", "local_elevation_250m", "local_elevation_500m",
    "log_flow_accumulation_m2", "height_above_drainage_m",
    "dem_relative_to_point_m", "imperviousness_pct",
    "topography_valid", "imperviousness_valid",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=REGIONS, default="santa-cruz-north")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    arguments = parser.parse_args()
    checkpoint_path = arguments.checkpoint if arguments.checkpoint.is_absolute() else ROOT / arguments.checkpoint
    dataset = ROOT / f"data/{arguments.region}-fema-map-class-terrain-sample.csv"
    patches_file = ROOT / f"data/cache/terrain-cnn/{arguments.region}-external-patches-float16.npy"
    metadata_file = ROOT / f"data/cache/terrain-cnn/{arguments.region}-external-patches.json"
    with dataset.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source)
                if row["mapped_hazard_class"] in {"minimal", "one_percent"}]
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata["datasetSha256"] != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise RuntimeError("CNN patch cache does not match the dataset")
    if metadata["channels"] != CHANNELS or [row["sample_id"] for row in metadata["rows"]] != [
        row["sample_id"] for row in rows
    ]:
        raise RuntimeError("CNN patch channels or row order do not match the dataset")
    patches = np.asarray(np.load(patches_file), dtype=np.float32)
    if patches.shape != (len(rows), 9, 64, 64) or not np.isfinite(patches).all():
        raise RuntimeError("Invalid CNN patch array")
    if not np.all(patches[:, 7, 32, 32] == 1):
        raise RuntimeError("CNN center topography is missing")
    raw = np.asarray([vector({**row, "location_id": row["sample_id"]}, row)
                      for row in rows], dtype=np.float32)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["channels"] != CHANNELS:
        raise RuntimeError("CNN checkpoint channels do not match")
    for channel in range(7):
        valid = patches[:, 7 if channel < 6 else 8] > 0.5
        values = patches[:, channel]
        values[valid] = np.clip(
            (values[valid] - checkpoint["pixelMean"][channel]) /
            checkpoint["pixelScale"][channel], -6, 6,
        )
    tabular = (np.where(np.isnan(raw), checkpoint["tabularMedian"], raw)
               - checkpoint["tabularMean"]) / checkpoint["tabularScale"]
    model = TerrainCNN()
    model.load_state_dict(checkpoint["stateDict"])
    model.eval()
    torch.set_num_threads(2)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            stop = start + 32
            logits = model(torch.from_numpy(patches[start:stop]),
                           torch.from_numpy(tabular[start:stop].astype(np.float32)))
            scores.extend(float(value) for value in torch.sigmoid(logits).numpy())
    print(json.dumps({"threshold": float(checkpoint["threshold"]), "scores": scores}))


if __name__ == "__main__":
    main()
