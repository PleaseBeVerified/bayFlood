"""Paired binary/three-class CNN coverage-weighting experiment.

Run with the trusted model environment and GPU access. Shared buffered splits,
training-only preprocessing/weights, and validation-only checkpoint selection.
Checkpoints stay in the ignored experiment cache; app models are never written.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from terrain_cnn_model import TerrainCNN
from terrain_features import FEATURES, joined_rows, vector
from terrain_coverage import coverage_weights
from terrain_three_class_metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/terrain-cnn"
EXPERIMENT = CACHE / "coverage-weighting"
REPORT = ROOT / "reports/terrain-cnn-coverage-weighting-pilot.json"
CLASSES = ["minimal", "one_percent", "point_two_percent"]
BATCH = 96
SEEDS = (42, 43, 44)
EPOCHS = {"binary": 14, "three_class": 12}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_stats(patches, indexes):
    sums, squares, counts = (np.zeros(7, dtype=float) for _ in range(3))
    for batch in np.array_split(indexes, max(1, math.ceil(len(indexes) / 128))):
        data = np.asarray(patches[batch], dtype=np.float32)
        for channel in range(7):
            valid = data[:, 7 if channel < 6 else 8] > .5
            values = data[:, channel][valid].astype(float)
            sums[channel] += values.sum()
            squares[channel] += np.square(values).sum()
            counts[channel] += len(values)
    assert (counts > 0).all()
    means = sums / counts
    scales = np.sqrt(np.maximum(squares / counts - means ** 2, 1e-6))
    return means.astype(np.float32), scales.astype(np.float32)


class Patches(Dataset):
    def __init__(self, patches, indexes, tabular, labels, weights=None):
        self.patches, self.indexes = patches, indexes
        self.tabular, self.labels = tabular, labels
        self.weights = weights if weights is not None else np.ones(len(labels), dtype=np.float32)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, position):
        i = int(self.indexes[position])
        return (torch.from_numpy(np.asarray(self.patches[i], dtype=np.float32)),
                torch.from_numpy(self.tabular[i]), int(self.labels[i]), float(self.weights[i]))


def normalize(patch, means, scales):
    valid = torch.cat((patch[:, 7:8].expand(-1, 6, -1, -1), patch[:, 8:9]), dim=1) > .5
    normalized = ((patch[:, :7] - means) / scales).clamp(-6, 6)
    return torch.cat((torch.where(valid, normalized, patch[:, :7]), patch[:, 7:]), dim=1)


@torch.inference_mode()
def predict(model, loader, device, means, scales, binary):
    model.eval()
    result = []
    for patch, values, _, _ in loader:
        logits = model(normalize(patch.to(device), means, scales), values.to(device))
        result.append((torch.sigmoid(logits) if binary else torch.softmax(logits, dim=1)).cpu().numpy())
    return np.concatenate(result)


def binary_metrics(actual, scores, threshold):
    predicted = scores >= threshold
    matrix = confusion_matrix(actual, predicted, labels=[0, 1]).tolist()
    return {
        "balancedAccuracy": float(balanced_accuracy_score(actual, predicted)),
        "macroF1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "onePercentF1": float(f1_score(actual, predicted, zero_division=0)),
        "rocAuc": float(roc_auc_score(actual, scores)),
        "brierScore": float(brier_score_loss(actual, scores)),
        "minimalFalsePositiveRate": float(predicted[actual == 0].mean()),
        "onePercentRecall": float(predicted[actual == 1].mean()),
        "confusionMatrix": matrix,
    }


def prepare(target, device):
    binary = target == "binary"
    rows, mapping = joined_rows(ROOT / "data/dnn-three-class-training-table.csv",
                                ROOT / "data/physical-hazard-calibration.csv", "mapped_hazard_class")
    metadata_path = CACHE / "three-class/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert [r["location_id"] for r in rows] == [r["location_id"] for r in metadata["rows"]]
    raw = np.asarray([mapping[r["location_id"]] for r in rows], dtype=np.float32)
    labels = np.asarray([CLASSES.index(r["mapped_hazard_class"]) for r in rows])
    assert np.array_equal(labels, [r["label"] for r in metadata["rows"]])
    xy = np.asarray([[r["x_m"], r["y_m"]] for r in metadata["rows"]])
    partition_path = CACHE / "three-class/partitions.json"
    partitions = {name: np.asarray(indexes, dtype=int) for name, indexes in json.loads(partition_path.read_text()).items()}
    if binary:
        partitions = {name: indexes[labels[indexes] != 2] for name, indexes in partitions.items()}
    for name, indexes in partitions.items():
        assert all(rows[i]["spatial_partition"] == name for i in indexes)
    train = partitions["training"]
    held = np.concatenate((partitions["validation"], partitions["test"]))
    assert not set(train) & set(held)
    assert cKDTree(xy[held]).query(xy[train], p=np.inf)[0].min() > 1920
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    scaler.fit(imputer.fit_transform(raw[train]))
    tabular = scaler.transform(imputer.transform(raw)).astype(np.float32)
    weights = np.ones(len(rows), dtype=np.float32)
    weights[train] = coverage_weights(labels[train], [rows[i]["county"] for i in train], raw[train], xy[train])
    saved_weights = {r["location_id"]: r for r in csv.DictReader((ROOT / "data/terrain-coverage-training-weights.csv").open())}
    column = "coverage_weight_binary" if binary else "coverage_weight_three_class"
    assert all(np.isclose(weights[i], float(saved_weights[rows[i]["location_id"]][column])) for i in train)
    patches = np.load(CACHE / "three-class/patches-float16.npy", mmap_mode="r")
    assert patches.shape == (len(rows), 9, 64, 64)
    mean, scale = pixel_stats(patches, train)
    means = torch.tensor(mean, device=device).reshape(1, 7, 1, 1)
    scales = torch.tensor(scale, device=device).reshape(1, 7, 1, 1)
    validation = DataLoader(Patches(patches, partitions["validation"], tabular, labels), batch_size=BATCH)
    evaluation = {
        "bay_test": (DataLoader(Patches(patches, partitions["test"], tabular, labels), batch_size=BATCH), labels[partitions["test"]]),
    }
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / "data/dnn-three-class-training-table.csv", ROOT / "data/physical-hazard-calibration.csv",
        partition_path, metadata_path, ROOT / "scripts/terrain_coverage.py",
        ROOT / "data/terrain-coverage-training-weights.csv",
    )}
    for region in ("santa-cruz-north", "tracy-lathrop-stockton"):
        dataset_path = ROOT / f"data/{region}-fema-map-class-terrain-sample.csv"
        samples = list(csv.DictReader(dataset_path.open()))
        assert {r["spatial_partition"] for r in samples} == {"external_test"}
        assert not {r["county"] for r in samples} & {r["county"] for r in rows}
        patch_metadata = json.loads((CACHE / f"three-class/{region}-external-patches.json").read_text())
        assert patch_metadata["datasetSha256"] == sha(dataset_path)
        assert patch_metadata["channels"] == metadata["channels"]
        assert [r["sample_id"] for r in samples] == [r["sample_id"] for r in patch_metadata["rows"]]
        external_labels = np.asarray([CLASSES.index(r["mapped_hazard_class"]) for r in samples])
        indexes = np.flatnonzero(external_labels != 2) if binary else np.arange(len(samples))
        external_raw = np.asarray([vector({**r, "location_id": r["sample_id"]}, r) for r in samples], dtype=np.float32)
        external_tabular = scaler.transform(imputer.transform(external_raw)).astype(np.float32)
        external_patches = np.load(CACHE / f"three-class/{region}-external-patches-float16.npy", mmap_mode="r")
        evaluation[region] = (DataLoader(Patches(external_patches, indexes, external_tabular, external_labels), batch_size=BATCH), external_labels[indexes])
        source_hashes[str(dataset_path.relative_to(ROOT))] = sha(dataset_path)
    return {
        "patches": patches, "partitions": partitions, "tabular": tabular, "labels": labels,
        "weights": weights, "means": means, "scales": scales, "validation": validation,
        "evaluation": evaluation, "sourceHashes": source_hashes,
        "preprocessing": {"pixelMean": mean, "pixelScale": scale,
                          "tabularMedian": imputer.statistics_, "tabularMean": scaler.mean_, "tabularScale": scaler.scale_},
    }


def train_one(data, target, variant, seed, device):
    binary = target == "binary"
    config = {"target": target, "variant": variant, "seed": seed, "epochs": EPOCHS[target],
              "batchSize": BATCH, "learningRate": .001, "weightDecay": .01,
              "classBalanced": not binary, "sourceHashes": data["sourceHashes"],
              "architectureSha256": sha(ROOT / "scripts/terrain_cnn_model.py"),
              "runnerSha256": sha(Path(__file__)), "torchVersion": str(torch.__version__), "device": str(device)}
    stem = EXPERIMENT / f"{target}-{variant}-seed{seed}"
    result_path = stem.with_suffix(".json")
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get("complete") and old.get("configuration") == config:
            print(f"Reusing completed {stem.name}", flush=True)
            return old
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    model = TerrainCNN(output_classes=1 if binary else 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    train = data["partitions"]["training"]
    labels = data["labels"]
    weights = data["weights"] if variant == "coverage" else np.ones(len(labels), dtype=np.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(Patches(data["patches"], train, data["tabular"], labels, weights),
                        batch_size=BATCH, shuffle=True, generator=generator, num_workers=0)
    class_weights = torch.ones(3, device=device)
    if not binary:
        class_weights = torch.tensor(len(train) / (3 * np.bincount(labels[train], minlength=3)), dtype=torch.float32, device=device)
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1, EPOCHS[target] + 1):
        epoch_started = time.monotonic()
        model.train()
        loss_sum = 0.0
        for patch, values, actual, weight in loader:
            patch, values, actual = patch.to(device), values.to(device), actual.to(device)
            weight = weight.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(normalize(patch, data["means"], data["scales"]), values)
            losses = nn.functional.binary_cross_entropy_with_logits(logits, actual.float(), reduction="none") if binary else nn.functional.cross_entropy(logits, actual, reduction="none") * class_weights[actual]
            loss = (losses * weight).mean()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(actual)
        scores = predict(model, data["validation"], device, data["means"], data["scales"], binary)
        actual = labels[data["partitions"]["validation"]]
        threshold = None
        if binary:
            positives = np.sort(scores[actual == 1])
            threshold = float(positives[len(positives) - math.ceil(.95 * len(positives))])
            evaluation = binary_metrics(actual, scores, threshold)
            rank = (-evaluation["minimalFalsePositiveRate"], -evaluation["brierScore"])
        else:
            evaluation = evaluate(actual, scores)
            rank = (evaluation["macroF1"], evaluation["perClass"]["point_two_percent"]["f1"], -evaluation["logLoss"])
        entry = {"epoch": epoch, "trainingLoss": loss_sum / len(train), "validation": evaluation,
                 "threshold": threshold, "seconds": time.monotonic() - epoch_started}
        history.append(entry)
        if best is None or rank > best[0]:
            best = (rank, entry)
            torch.save({"stateDict": {key: value.cpu().clone() for key, value in model.state_dict().items()},
                        "configuration": config, "preprocessing": data["preprocessing"],
                        "features": FEATURES, "threshold": threshold, "classes": CLASSES[:2] if binary else CLASSES}, stem.with_suffix(".pt"))
        result_path.write_text(json.dumps({"complete": False, "configuration": config, "history": history}, indent=2) + "\n")
        score_label = f"FPR={evaluation['minimalFalsePositiveRate']:.3f}" if binary else f"macroF1={evaluation['macroF1']:.3f}"
        print(f"{target} {variant} seed={seed} epoch={epoch}/{EPOCHS[target]} "
              f"loss={entry['trainingLoss']:.3f} val {score_label} {entry['seconds']:.1f}s", flush=True)
    checkpoint = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["stateDict"])
    evaluation = {}
    for region, (loader, actual) in data["evaluation"].items():
        scores = predict(model, loader, device, data["means"], data["scales"], binary)
        evaluation[region] = binary_metrics(actual, scores, checkpoint["threshold"]) if binary else evaluate(actual, scores)
    result = {"complete": True, "configuration": config, "history": history,
              "selectedEpoch": best[1]["epoch"], "selectedValidation": best[1]["validation"],
              "threshold": checkpoint["threshold"], "evaluation": evaluation,
              "trainingSeconds": time.monotonic() - started,
              "partitionCounts": {k: len(v) for k, v in data["partitions"].items()},
              "checkpoint": str(stem.with_suffix(".pt").relative_to(ROOT))}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Completed {stem.name}: " + ", ".join(f"{region} BA={m['balancedAccuracy']:.3f}" for region, m in evaluation.items()), flush=True)
    return result


def write_report(runs):
    aggregate = {}
    for target in ("binary", "three_class"):
        picks = [r for r in runs if r["configuration"]["target"] == target]
        if not picks:
            continue
        aggregate[target] = {}
        for region in ("bay_test", "santa-cruz-north", "tracy-lathrop-stockton"):
            aggregate[target][region] = {}
            for variant in ("control", "coverage"):
                values = [r["evaluation"][region] for r in picks if r["configuration"]["variant"] == variant]
                if not values:
                    continue
                matrices = np.asarray([v["confusionMatrix"] for v in values], dtype=float)
                aggregate[target][region][variant] = {
                    "runCount": len(values), "balancedAccuracyMean": float(np.mean([v["balancedAccuracy"] for v in values])),
                    "balancedAccuracyRange": [float(min(v["balancedAccuracy"] for v in values)), float(max(v["balancedAccuracy"] for v in values))],
                    "macroF1Mean": float(np.mean([v["macroF1"] for v in values])),
                    "meanConfusionMatrix": matrices.mean(axis=0).tolist(),
                    "perClassRecall": dict(zip(CLASSES, (np.diagonal(matrices, axis1=1, axis2=2) / matrices.sum(axis=2)).mean(axis=0).tolist())),
                }
    report = {"complete": len(runs) == 12, "generatedAt": datetime.now(timezone.utc).isoformat(),
              "protocol": "Three paired seeds per binary/three-class target. Same buffered IDs, architecture, initialization seed, minibatch order, preprocessing and epoch budget within pairs; only terrain-coverage loss weights differ. Binary: 14 epochs, Bay validation checkpoint chosen by lowest FPR at >=95% recall, then Brier. Three-class: 12 epochs with class-balanced loss in both variants; Bay validation selects macro F1, then 0.2% F1, then log loss. External regions scored only after checkpoint selection.",
              "limitations": "Exploratory reuse of development regions; three seeds are not independent geographic replications. MPS training is seeded but not guaranteed bitwise deterministic. These paired controls use shared buffers and are not the earlier saved checkpoints. No new terrain examples acquired and no app model replaced.",
              "runs": runs, "aggregate": aggregate}
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# CNN training coverage weighting pilot", "", report["protocol"], "",
             f"Completed {len(runs)}/12 target/variant/seed runs.", "",
             "Mean balanced accuracy across completed seeds, control → coverage weighted:", "",
             "| CNN | Bay test | Santa Cruz | Tracy–Lathrop–Stockton |", "| --- | ---: | ---: | ---: |"]
    for target, regions in aggregate.items():
        if any(len(regions[r]) != 2 for r in regions):
            continue
        cells = [f"{regions[r]['control']['balancedAccuracyMean']:.3f} → {regions[r]['coverage']['balancedAccuracyMean']:.3f}"
                 for r in ("bay_test", "santa-cruz-north", "tracy-lathrop-stockton")]
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    lines += ["", "Class-specific rates, control → coverage weighted:", "",
              "| Target / region | Minimal false-positive rate | Mapped 1% recall | Mapped 0.2% recall |",
              "| --- | ---: | ---: | ---: |"]
    for target, regions in aggregate.items():
        for region, variants in regions.items():
            if len(variants) != 2:
                continue
            left, right = variants["control"], variants["coverage"]
            fpr = f"{1-left['perClassRecall']['minimal']:.3f} → {1-right['perClassRecall']['minimal']:.3f}"
            one = f"{left['perClassRecall']['one_percent']:.3f} → {right['perClassRecall']['one_percent']:.3f}"
            two = "—" if target == "binary" else f"{left['perClassRecall']['point_two_percent']:.3f} → {right['perClassRecall']['point_two_percent']:.3f}"
            lines.append(f"| {target} / {region} | {fpr} | {one} | {two} |")
    lines += ["", "Individual seed confusion matrices, score metrics, and selected epochs are in the adjacent JSON.",
              "Class order is minimal, 1%, 0.2% (binary excludes 0.2%).", "", report["limitations"], "",
              "Reproduce: `.venv-model/bin/python scripts/train-terrain-cnn-coverage-pilot.py`.",
              "Completed matching cache runs are reused; new configurations train from scratch.", ""]
    REPORT.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("all", "binary", "three_class"), default="all")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    print(f"CNN coverage experiment on {device}; three seeds; 14 binary / 12 three-class epochs", flush=True)
    runs = []
    for target in ("binary", "three_class") if arguments.target == "all" else (arguments.target,):
        data = prepare(target, device)
        print(f"Prepared {target}: " + str({k: len(v) for k, v in data["partitions"].items()}), flush=True)
        for seed in SEEDS:
            for variant in ("control", "coverage"):
                runs.append(train_one(data, target, variant, seed, device))
                write_report(runs)
        del data
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
