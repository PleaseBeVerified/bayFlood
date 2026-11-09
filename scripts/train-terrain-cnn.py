#!/usr/bin/env python3
"""Offline point-centered 2D CNN experiment for FEMA map-class agreement."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from terrain_features import joined_rows
from terrain_cnn_model import TerrainCNN


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/terrain-cnn"
REPORT = ROOT / "reports/terrain-cnn-evaluation.json"
REPORT_MD = ROOT / "reports/terrain-cnn-evaluation.md"
MODEL = ROOT / "models/flood-risk-terrain-cnn.pt"
SEED = 42
TARGET_RECALL = 0.95
BUFFER_METERS = 1920  # A 64 x 64 patch at 30 m spans 1,920 m on each axis.
EPOCHS = 14
BATCH_SIZE = 96


def threshold_for_recall(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = np.sort(scores[labels == 1])
    required = math.ceil(TARGET_RECALL * len(positive))
    return float(positive[len(positive) - required])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    negatives = labels == 0
    positives = labels == 1
    return {
        "rows": int(len(labels)),
        "minimalRows": int(negatives.sum()),
        "onePercentRows": int(positives.sum()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predicted)),
        "minimalFalsePositiveRate": float(predicted[negatives].mean()),
        "minimalFalsePositives": int(predicted[negatives].sum()),
        "onePercentRecall": float(recall_score(labels, predicted)),
        "onePercentPrecision": float(precision_score(labels, predicted, zero_division=0)),
        "rocAuc": float(roc_auc_score(labels, scores)),
        "brierScore": float(brier_score_loss(labels, scores)),
        "confusionMatrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def matched_test_recall(labels: np.ndarray, scores: np.ndarray, detected: int) -> dict:
    """Post-hoc ranking diagnostic, never used to choose an operating threshold."""
    positive = np.sort(scores[labels == 1])
    threshold = float(positive[len(positive) - detected])
    predicted = scores >= threshold
    return {
        "thresholdChosenOnTest": threshold,
        "onePercentDetected": int(predicted[labels == 1].sum()),
        "minimalFalsePositives": int(predicted[labels == 0].sum()),
        "minimalFalsePositiveRate": float(predicted[labels == 0].mean()),
    }


def buffered_partitions(metadata: list[dict]) -> dict[str, np.ndarray]:
    partitions = {
        name: np.array([i for i, row in enumerate(metadata) if row["partition"] == name])
        for name in ("training", "validation", "test")
    }
    xy = np.array([[row["x_m"], row["y_m"]] for row in metadata])
    test_tree = cKDTree(xy[partitions["test"]])
    validation = partitions["validation"]
    validation = validation[np.isinf(test_tree.query(xy[validation], p=np.inf,
                                                   distance_upper_bound=BUFFER_METERS)[0])]
    holdout_tree = cKDTree(xy[np.concatenate((validation, partitions["test"]))])
    training = partitions["training"]
    training = training[np.isinf(holdout_tree.query(xy[training], p=np.inf,
                                                  distance_upper_bound=BUFFER_METERS)[0])]
    if not len(training) or not len(validation):
        raise RuntimeError("Buffer removed a full partition")
    return {"training": training, "validation": validation, "test": partitions["test"]}


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
    means = sums / counts
    scales = np.sqrt(np.maximum(squares / counts - means * means, 1e-6))
    return means.astype(np.float32), scales.astype(np.float32)


class PatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, indices: np.ndarray, tabular: np.ndarray,
                 labels: np.ndarray, elevation: np.ndarray, means: np.ndarray,
                 scales: np.ndarray, hard_minimal: bool):
        self.patches, self.indices = patches, indices
        self.tabular, self.labels, self.elevation = tabular, labels, elevation
        self.means, self.scales, self.hard_minimal = means, scales, hard_minimal

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
        weight = 1.5 if self.hard_minimal and self.labels[index] == 0 and self.elevation[index] < 50 else 1.0
        return (
            torch.from_numpy(data), torch.from_numpy(self.tabular[index]),
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


@torch.no_grad()
def predict(model: TerrainCNN, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for patches, values, _, _ in loader:
        logits = model(patches.to(device), values.to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true",
                        help="Run a lower-learning-rate experiment with early stopping and separate artifacts")
    args = parser.parse_args()
    max_epochs = 80 if args.extended else EPOCHS
    learning_rate = 3e-4 if args.extended else 1e-3
    patience = 15 if args.extended else None
    report_path = (ROOT / "reports/terrain-cnn-extended-evaluation.json") if args.extended else REPORT
    report_md_path = (ROOT / "reports/terrain-cnn-extended-evaluation.md") if args.extended else REPORT_MD
    model_path = (ROOT / "models/flood-risk-terrain-cnn-extended.pt") if args.extended else MODEL
    scores_path = CACHE / ("extended-selected-test-scores.npy" if args.extended else "selected-test-scores.npy")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    metadata = json.loads((CACHE / "metadata.json").read_text())
    rows = metadata["rows"]
    patches = np.load(CACHE / "patches-float16.npy", mmap_mode="r")
    table, x_by_id = joined_rows(ROOT / "data/dnn-training-table.csv",
                                  ROOT / "data/physical-hazard-calibration.csv")
    if [item["location_id"] for item in rows] != [item["location_id"] for item in table]:
        raise RuntimeError("Patch cache and training table row order differ")
    indices = buffered_partitions(rows)
    raw = np.asarray([x_by_id[row["location_id"]] for row in rows], dtype=np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    elevation = np.asarray([row["elevation_m"] for row in rows], dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    tabular = np.empty_like(raw)
    tabular[indices["training"]] = scaler.fit_transform(imputer.fit_transform(raw[indices["training"]]))
    for split in ("validation", "test"):
        tabular[indices[split]] = scaler.transform(imputer.transform(raw[indices[split]]))
    means, scales = channel_stats(patches, indices["training"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}; buffered rows: " + ", ".join(
        f"{name}={len(part)}" for name, part in indices.items()), flush=True)
    report = {
        "schemaVersion": 1, "generatedAt": datetime.now(timezone.utc).isoformat(),
        "target": "FEMA mapped one_percent versus minimal; not observed storm inundation",
        "patch": {"size": 64, "resolutionMeters": 30, "channels": metadata["channels"]},
        "bufferMeters": BUFFER_METERS, "targetValidationRecall": TARGET_RECALL,
        "partitionCounts": {name: len(part) for name, part in indices.items()},
        "partitionLabels": {name: {"minimal": int((labels[part] == 0).sum()),
                                   "one_percent": int((labels[part] == 1).sum())}
                            for name, part in indices.items()},
        "pixelMean": means.tolist(), "pixelScale": scales.tolist(),
        "training": {"maxEpochs": max_epochs, "learningRate": learning_rate,
                     "earlyStoppingPatience": patience, "weightDecay": 0.01},
        "device": str(device), "candidates": [],
    }
    # XGBoost and PyTorch can load conflicting native runtimes on macOS.
    subprocess.run([sys.executable, str(ROOT / "scripts/terrain-cnn-baseline.py")], check=True)
    baseline = json.loads((CACHE / "baseline.json").read_text())
    for name in indices:
        if indices[name].tolist() != baseline[f"{name}Indices"]:
            raise RuntimeError(f"Baseline {name} indices differ from CNN indices")
    baseline_threshold = baseline["validationThreshold"]
    baseline_test = np.asarray(baseline["testScores"])
    report["bufferedTabularBaseline"] = {
        "validationThreshold": baseline_threshold,
        "validation": baseline["validation"],
        "test": baseline["test"],
    }
    loaders = {}
    for name in ("validation", "test"):
        dataset = PatchDataset(patches, indices[name], tabular, labels, elevation, means, scales, False)
        loaders[name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    best = None
    for hard_minimal in (False, True):
        torch.manual_seed(SEED)
        model = TerrainCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        dataset = PatchDataset(patches, indices["training"], tabular, labels, elevation,
                               means, scales, hard_minimal)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        history = []
        candidate_best = None
        epochs_without_improvement = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            running = 0.0
            for patch, values, target, weight in loader:
                patch, values = patch.to(device), values.to(device)
                target, weight = target.to(device), weight.to(device)
                optimizer.zero_grad()
                loss = (nn.functional.binary_cross_entropy_with_logits(
                    model(patch, values), target, reduction="none") * weight).mean()
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * len(target)
            scores = predict(model, loaders["validation"], device)
            threshold = threshold_for_recall(labels[indices["validation"]], scores)
            evaluation = metrics(labels[indices["validation"]], scores, threshold)
            candidate = {
                "hardMinimal": hard_minimal, "epoch": epoch, "trainingLoss": running / len(dataset),
                "validationThreshold": threshold, "validation": evaluation,
            }
            history.append(candidate)
            # Selection uses validation only; minimize false alarms at fixed recall.
            rank = (evaluation["minimalFalsePositiveRate"], evaluation["brierScore"])
            if candidate_best is None or rank < candidate_best:
                candidate_best = rank
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if best is None or rank < best[0]:
                best = (rank, candidate, {key: value.cpu().clone()
                                         for key, value in model.state_dict().items()})
            print(f"hard_minimal={hard_minimal} epoch={epoch} loss={candidate['trainingLoss']:.4f} "
                  f"val FPR={evaluation['minimalFalsePositiveRate']:.3f} "
                  f"recall={evaluation['onePercentRecall']:.3f} "
                  f"Brier={evaluation['brierScore']:.3f}", flush=True)
            if patience is not None and epochs_without_improvement >= patience:
                print(f"Early stopping hard_minimal={hard_minimal} after epoch {epoch}", flush=True)
                break
        report["candidates"].append({"hardMinimal": hard_minimal, "history": history,
                                     "stoppedAfterEpoch": history[-1]["epoch"]})
    assert best is not None
    chosen = TerrainCNN().to(device)
    chosen.load_state_dict(best[2])
    test_scores = predict(chosen, loaders["test"], device)
    np.save(scores_path, test_scores)
    selected = best[1]
    report["selected"] = {
        "hardMinimal": selected["hardMinimal"], "epoch": selected["epoch"],
        "validationThreshold": selected["validationThreshold"],
        "validation": selected["validation"],
        "test": metrics(labels[indices["test"]], test_scores, selected["validationThreshold"]),
    }
    test_labels = labels[indices["test"]]
    detected = report["selected"]["test"]["confusionMatrix"][1][1]
    report["matchedTestRecallDiagnostic"] = {
        "meaning": "Post-hoc test-label thresholds for ranking comparison only; not deployable thresholds",
        "onePercentDetectedTarget": detected,
        "baseline": matched_test_recall(test_labels, baseline_test, detected),
        "cnn": matched_test_recall(test_labels, test_scores, detected),
    }
    report["testMinimalElevationBands"] = {}
    for name, bounds in {"below_10m": (-math.inf, 10), "10_to_50m": (10, 50),
                         "50_to_282m": (50, 282.1), "at_least_282m": (282.1, math.inf)}.items():
        picks = np.where((labels[indices["test"]] == 0) &
                         (elevation[indices["test"]] >= bounds[0]) &
                         (elevation[indices["test"]] < bounds[1]))[0]
        report["testMinimalElevationBands"][name] = {
            "rows": len(picks),
            "cnnFalsePositives": int((test_scores[picks] >= selected["validationThreshold"]).sum()),
            "baselineFalsePositives": int((baseline_test[picks] >= baseline_threshold).sum()),
        }
    torch.save({"stateDict": best[2], "pixelMean": means, "pixelScale": scales,
                "tabularMedian": imputer.statistics_, "tabularMean": scaler.mean_,
                "tabularScale": scaler.scale_, "threshold": selected["validationThreshold"],
                "channels": metadata["channels"], "target": report["target"]}, model_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    base = report["bufferedTabularBaseline"]["test"]
    cnn = report["selected"]["test"]
    lines = [
        "# Point-centered terrain CNN FEMA map-class experiment" +
        (" — extended training" if args.extended else ""), "",
        "This offline model predicts FEMA `one_percent` versus `minimal` map labels, not observed flooding. "
        "Each GPS coordinate centers a 64 × 64 grid of 30 m raster pixels. "
        "A 1,920 m Chebyshev buffer removes training locations whose patches could overlap a "
        "validation or test patch; validation patches near test patches are also removed.", "",
        f"Buffered rows: training {len(indices['training'])}, validation {len(indices['validation'])}, "
        f"test {len(indices['test'])}. Model selection and the 95% recall threshold use validation only.", "",
        f"Training: up to {max_epochs} epochs, AdamW learning rate {learning_rate:g}, "
        f"early-stopping patience {patience if patience is not None else 'none'}; "
        f"selected {'weighted' if selected['hardMinimal'] else 'unweighted'} epoch {selected['epoch']}.", "",
        "| Test metric | Buffered terrain XGBoost | CNN + point features |",
        "| --- | ---: | ---: |",
    ]
    for label, key in [("Minimal false positives", "minimalFalsePositives"),
                       ("Minimal false-positive rate", "minimalFalsePositiveRate"),
                       ("1% recall", "onePercentRecall"),
                       ("Balanced accuracy", "balancedAccuracy"),
                       ("ROC AUC", "rocAuc"), ("Brier score", "brierScore")]:
        fmt = ".0f" if key == "minimalFalsePositives" else ".3f"
        lines.append(f"| {label} | {base[key]:{fmt}} | {cnn[key]:{fmt}} |")
    lines += ["", "Minimal-class false positives by elevation:", "",
              "| Elevation | Rows | XGBoost errors | CNN errors |", "| --- | ---: | ---: | ---: |"]
    for name, values in report["testMinimalElevationBands"].items():
        lines.append(f"| {name.replace('_', ' ')} | {values['rows']} | "
                     f"{values['baselineFalsePositives']} | {values['cnnFalsePositives']} |")
    matched = report["matchedTestRecallDiagnostic"]
    lines += ["", f"At the same **test** recall ({detected}/{int((test_labels == 1).sum())}), "
              f"a post-hoc ranking check gives {matched['baseline']['minimalFalsePositives']} "
              f"XGBoost versus {matched['cnn']['minimalFalsePositives']} CNN `minimal` false positives. "
              "These test-derived thresholds are diagnostic only and cannot be used for deployment."]
    lines += ["", "The same geographic test partition has been used for earlier model studies, "
              "so it is a diagnostic comparison rather than fresh independent confirmation. "
              "Neither model was evaluated against observed inundation at FEMA-minimal locations.", ""]
    report_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected hard_minimal={selected['hardMinimal']} epoch={selected['epoch']}; "
          f"test Minimal FPR baseline={base['minimalFalsePositiveRate']:.3f}, "
          f"CNN={cnn['minimalFalsePositiveRate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
