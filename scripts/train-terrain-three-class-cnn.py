#!/usr/bin/env python3
"""Offline three-class terrain CNN experiment using all FEMA map classes."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from terrain_cnn_model import TerrainCNN
from terrain_features import FEATURES, joined_rows
from terrain_three_class_metrics import CLASSES, evaluate


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/terrain-cnn/three-class"
REPORT = ROOT / "reports/terrain-three-class-cnn-evaluation.json"
REPORT_MD = ROOT / "reports/terrain-three-class-cnn-evaluation.md"
MODEL = ROOT / "models/flood-hazard-terrain-three-class-cnn.pt"
SEED = 42
BUFFER_METERS = 1920
EPOCHS = 12
BATCH_SIZE = 96


def buffered_partitions(rows: list[dict]) -> dict[str, np.ndarray]:
    partitions = {
        name: np.asarray([i for i, row in enumerate(rows) if row["partition"] == name],
                         dtype=np.int64)
        for name in ("training", "validation", "test")
    }
    xy = np.asarray([[row["x_m"], row["y_m"]] for row in rows])
    test_tree = cKDTree(xy[partitions["test"]])
    validation = partitions["validation"]
    validation = validation[np.isinf(test_tree.query(
        xy[validation], p=np.inf, distance_upper_bound=BUFFER_METERS,
    )[0])]
    holdout_tree = cKDTree(xy[np.concatenate((validation, partitions["test"]))])
    training = partitions["training"]
    training = training[np.isinf(holdout_tree.query(
        xy[training], p=np.inf, distance_upper_bound=BUFFER_METERS,
    )[0])]
    result = {"training": training, "validation": validation, "test": partitions["test"]}
    if any(len(part) == 0 for part in result.values()):
        raise RuntimeError("The patch overlap buffer removed a whole partition")
    return result


def channel_stats(patches: np.ndarray, training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(7, dtype=np.float64)
    squares = np.zeros(7, dtype=np.float64)
    counts = np.zeros(7, dtype=np.float64)
    for batch in np.array_split(training, max(1, math.ceil(len(training) / 128))):
        data = np.asarray(patches[batch], dtype=np.float32)
        for channel in range(7):
            valid = data[:, 7 if channel < 6 else 8] > 0.5
            values = data[:, channel][valid].astype(np.float64)
            sums[channel] += values.sum()
            squares[channel] += np.square(values).sum()
            counts[channel] += len(values)
    if np.any(counts == 0):
        raise RuntimeError("A CNN image channel has no valid training pixels")
    means = sums / counts
    scales = np.sqrt(np.maximum(squares / counts - means * means, 1e-6))
    return means.astype(np.float32), scales.astype(np.float32)


class PatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, indices: np.ndarray, tabular: np.ndarray,
                 labels: np.ndarray, means: np.ndarray, scales: np.ndarray):
        self.patches, self.indices, self.tabular, self.labels = patches, indices, tabular, labels
        self.means, self.scales = means, scales

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        data = np.asarray(self.patches[index], dtype=np.float32).copy()
        for channel in range(7):
            valid = data[7 if channel < 6 else 8] > 0.5
            data[channel, valid] = np.clip(
                (data[channel, valid] - self.means[channel]) / self.scales[channel], -6, 6,
            )
        return (torch.from_numpy(data), torch.from_numpy(self.tabular[index]),
                torch.tensor(self.labels[index], dtype=torch.long))


@torch.no_grad()
def predict(model: TerrainCNN, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    for patches, values, _ in loader:
        logits = model(patches.to(device), values.to(device))
        batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(batches)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    rows = metadata["rows"]
    if metadata.get("classes") != list(CLASSES):
        raise RuntimeError("Unexpected CNN patch class order")
    patches = np.load(CACHE / "patches-float16.npy", mmap_mode="r")
    if patches.shape != (len(rows), 9, 64, 64):
        raise RuntimeError("Unexpected three-class patch array shape")
    table, vectors = joined_rows(
        ROOT / "data/dnn-three-class-training-table.csv",
        ROOT / "data/physical-hazard-calibration.csv", "mapped_hazard_class",
    )
    if [r["location_id"] for r in rows] != [r["location_id"] for r in table]:
        raise RuntimeError("CNN patch cache order differs from the three-class table")
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    if any(CLASSES[label] != row["mapped_hazard_class"]
           for label, row in zip(labels, table, strict=True)):
        raise RuntimeError("CNN patch labels disagree with the source table")
    indices = buffered_partitions(rows)
    if any(set(labels[part]) != {0, 1, 2} for part in indices.values()):
        raise RuntimeError("A buffered partition is missing a map class")
    (CACHE / "partitions.json").write_text(json.dumps(
        {name: part.tolist() for name, part in indices.items()},
    ), encoding="utf-8")
    raw = np.asarray([vectors[r["location_id"]] for r in rows], dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    tabular = np.empty_like(raw)
    tabular[indices["training"]] = scaler.fit_transform(
        imputer.fit_transform(raw[indices["training"]]),
    )
    for split in ("validation", "test"):
        tabular[indices[split]] = scaler.transform(imputer.transform(raw[indices[split]]))
    means, scales = channel_stats(patches, indices["training"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    label_counts = {
        name: {label: int((labels[part] == i).sum()) for i, label in enumerate(CLASSES)}
        for name, part in indices.items()
    }
    print(f"Device: {device}; buffered rows: " + ", ".join(
        f"{name}={len(part)}" for name, part in indices.items()), flush=True)

    # XGBoost and PyTorch use conflicting native runtimes on some macOS installs.
    subprocess.run([sys.executable, str(ROOT / "scripts/terrain-three-class-cnn-baseline.py")],
                   cwd=ROOT, check=True)
    baseline = json.loads((CACHE / "baseline.json").read_text(encoding="utf-8"))
    loaders = {}
    for name in ("validation", "test"):
        dataset = PatchDataset(patches, indices[name], tabular, labels, means, scales)
        loaders[name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    candidates = []
    best = None
    counts = np.bincount(labels[indices["training"]], minlength=3)
    class_weights = torch.tensor(len(indices["training"]) / (3 * counts),
                                 dtype=torch.float32, device=device)
    for balanced in (False, True):
        torch.manual_seed(SEED)
        model = TerrainCNN(output_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights if balanced else None)
        dataset = PatchDataset(patches, indices["training"], tabular, labels, means, scales)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        history = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            for patch, values, target in loader:
                patch, values, target = patch.to(device), values.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(patch, values), target)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(target)
            validation = evaluate(labels[indices["validation"]],
                                  predict(model, loaders["validation"], device))
            candidate = {"classWeight": "balanced" if balanced else "none",
                         "epoch": epoch, "trainingLoss": loss_sum / len(dataset),
                         "validation": validation}
            history.append(candidate)
            rank = (validation["macroF1"],
                    validation["perClass"]["point_two_percent"]["f1"],
                    -validation["logLoss"])
            if best is None or rank > best[0]:
                best = (rank, candidate, {key: value.cpu().clone()
                                         for key, value in model.state_dict().items()})
            print(f"CNN balanced={balanced} epoch={epoch} "
                  f"loss={candidate['trainingLoss']:.3f} "
                  f"val macro F1={validation['macroF1']:.3f} "
                  f"0.2% F1={validation['perClass']['point_two_percent']['f1']:.3f}",
                  flush=True)
        candidates.append({"classWeight": "balanced" if balanced else "none",
                           "history": history})
    assert best is not None
    chosen = TerrainCNN(output_classes=3).to(device)
    chosen.load_state_dict(best[2])
    test = evaluate(labels[indices["test"]], predict(chosen, loaders["test"], device))
    selected = best[1]
    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "target": "FEMA map class, not observed inundation or property flood probability",
        "classes": CLASSES,
        "inputFiles": ["data/dnn-three-class-training-table.csv",
                       "data/physical-hazard-calibration.csv"],
        "pointFeatures": FEATURES,
        "patch": {"size": 64, "resolutionMeters": 30, "channels": metadata["channels"]},
        "bufferMeters": BUFFER_METERS,
        "partitionCounts": {name: len(part) for name, part in indices.items()},
        "labelCounts": label_counts,
        "selectionCriterion": "highest buffered geographic-validation macro F1, then 0.2% F1, then log loss",
        "candidates": candidates,
        "selected": {"classWeight": selected["classWeight"], "epoch": selected["epoch"],
                     "validation": selected["validation"], "test": test},
        "boostedComparator": baseline,
    }
    torch.save({
        "state_dict": best[2], "classes": CLASSES, "point_features": FEATURES,
        "patch_channels": metadata["channels"], "pixel_mean": means,
        "pixel_scale": scales, "imputer_statistics": imputer.statistics_,
        "point_mean": scaler.mean_, "point_scale": scaler.scale_,
    }, MODEL)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Three-class terrain CNN experiment", "",
        "The CNN uses 64 × 64 terrain and imperviousness patches plus ten point inputs.",
        "All 933 mapped 0.2% locations are eligible. A 1,920 m buffer removes training",
        "and validation patches that could overlap a held-out patch. The boosted",
        "comparator uses the same buffered locations and ten point inputs.", "",
        f"Buffered rows: training {len(indices['training'])}, validation "
        f"{len(indices['validation'])}, test {len(indices['test'])}.",
        f"Selected CNN: {selected['classWeight']} class weight, epoch {selected['epoch']}, "
        "chosen on validation only.", "",
        "| Geographic test metric | CNN | Buffered terrain boosted |",
        "| --- | ---: | ---: |",
    ]
    boosted_test = baseline["test"]
    for label, key in (("Accuracy", "accuracy"), ("Balanced accuracy", "balancedAccuracy"),
                       ("Macro F1", "macroF1"), ("Log loss", "logLoss")):
        lines.append(f"| {label} | {test[key]:.3f} | {boosted_test[key]:.3f} |")
    lines += ["", "| Class | Test samples | CNN precision | CNN recall | CNN F1 | Boosted F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in CLASSES:
        current = test["perClass"][name]
        comparison = boosted_test["perClass"][name]
        lines.append(f"| {name} | {current['support']} | {current['precision']:.3f} | "
                     f"{current['recall']:.3f} | {current['f1']:.3f} | {comparison['f1']:.3f} |")
    lines += ["", f"CNN confusion matrix (actual rows, predicted columns; class order "
              f"{', '.join(CLASSES)}): `{test['confusionMatrix']}`.",
              f"Boosted confusion matrix: `{boosted_test['confusionMatrix']}`.", "",
              "These scores measure agreement with FEMA map classes, not observed floods.",
              "The test blocks have been used in earlier model comparisons, so this is",
              "a diagnostic comparison rather than a fresh independent confirmation.",
              "Levee-related or otherwise unavailable map categories are not labels here.", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"CNN test macro F1 {test['macroF1']:.3f}; "
          f"0.2% F1 {test['perClass']['point_two_percent']['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
