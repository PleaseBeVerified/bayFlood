#!/usr/bin/env python3
"""Score a three-class CNN external holdout in an isolated PyTorch process."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from terrain_cnn_model import TerrainCNN
from terrain_features import FEATURES, vector
from terrain_three_class_metrics import CLASSES


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/terrain-cnn/three-class"
CHECKPOINT = ROOT / "models/flood-hazard-terrain-three-class-cnn.pt"
REGIONS = ("santa-cruz-north", "tracy-lathrop-stockton")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=REGIONS, default="santa-cruz-north")
    arguments = parser.parse_args()
    dataset = ROOT / f"data/{arguments.region}-fema-map-class-terrain-sample.csv"
    metadata_file = CACHE / f"{arguments.region}-external-patches.json"
    patch_file = CACHE / f"{arguments.region}-external-patches-float16.npy"
    with dataset.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    metadata = json.loads(metadata_file.read_text())
    if metadata["datasetSha256"] != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise RuntimeError("External patch cache does not match the source CSV")
    if [r["sample_id"] for r in metadata["rows"]] != [r["sample_id"] for r in rows]:
        raise RuntimeError("External patch cache order differs from the source CSV")
    patches = np.asarray(np.load(patch_file),
                         dtype=np.float32)
    if patches.shape != (len(rows), 9, 64, 64) or not np.isfinite(patches).all():
        raise RuntimeError("Unexpected external patch array")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if list(checkpoint["classes"]) != list(CLASSES) or checkpoint["point_features"] != FEATURES:
        raise RuntimeError("Checkpoint label or point feature contract differs")
    if checkpoint["patch_channels"] != metadata["channels"]:
        raise RuntimeError("Checkpoint image channel order differs")
    raw = np.asarray([vector({**r, "location_id": r["sample_id"]}, r)
                      for r in rows], dtype=np.float32)
    median = np.asarray(checkpoint["imputer_statistics"], dtype=np.float32)
    mean = np.asarray(checkpoint["point_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["point_scale"], dtype=np.float32)
    tabular = (np.where(np.isnan(raw), median, raw) - mean) / scale
    if not np.isfinite(tabular).all():
        raise RuntimeError("External point inputs contain missing values")
    for channel in range(7):
        valid = patches[:, 7 if channel < 6 else 8] > 0.5
        values = patches[:, channel]
        values[valid] = np.clip(
            (values[valid] - checkpoint["pixel_mean"][channel]) /
            checkpoint["pixel_scale"][channel], -6, 6,
        )
    model = TerrainCNN(output_classes=3)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(4)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            logits = model(torch.from_numpy(patches[start:start + 32]),
                           torch.from_numpy(tabular[start:start + 32].astype(np.float32)))
            probabilities.extend(torch.softmax(logits, dim=1).numpy().tolist())
    print(json.dumps({"probabilities": probabilities}))


if __name__ == "__main__":
    main()
