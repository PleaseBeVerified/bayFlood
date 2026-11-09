#!/usr/bin/env python3
"""Test whether the terrain CNN's ten point inputs work without image pixels."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from terrain_features import FEATURES, joined_rows


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/terrain-cnn"
REPORT = ROOT / "reports/terrain-cnn-point-only-ablation.json"
REPORT_MD = ROOT / "reports/terrain-cnn-point-only-ablation.md"
SEED = 42
EPOCHS = 100
BATCH_SIZE = 96
TARGET_RECALL = 0.95


class PointOnlyHead(nn.Module):
    """The CNN's tabular branch and output head, with the image branch removed."""

    def __init__(self) -> None:
        super().__init__()
        self.tabular = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.tabular(values)).squeeze(1)


def threshold_for_recall(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = np.sort(scores[labels == 1])
    required = math.ceil(TARGET_RECALL * len(positives))
    return float(positives[len(positives) - required])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    negatives = labels == 0
    return {
        "rows": int(len(labels)),
        "minimalRows": int(negatives.sum()),
        "onePercentRows": int((labels == 1).sum()),
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


@torch.no_grad()
def predict(model: PointOnlyHead, values: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(values))).numpy()


def matched_recall_errors(labels: np.ndarray, scores: np.ndarray, detections: int) -> int:
    """Test-label diagnostic only; this threshold must not be deployed."""
    positives = np.sort(scores[labels == 1])
    threshold = positives[len(positives) - detections]
    return int(((scores >= threshold) & (labels == 0)).sum())


def main() -> None:
    torch.set_num_threads(4)
    metadata = json.loads((CACHE / "metadata.json").read_text())
    rows = metadata["rows"]
    table, features_by_id = joined_rows(ROOT / "data/dnn-training-table.csv",
                                        ROOT / "data/physical-hazard-calibration.csv")
    if [row["location_id"] for row in rows] != [row["location_id"] for row in table]:
        raise RuntimeError("Point data and CNN patch cache have different row orders")
    baseline = json.loads((CACHE / "baseline.json").read_text())
    cnn_report = json.loads((ROOT / "reports/terrain-cnn-evaluation.json").read_text())
    indices = {split: np.asarray(baseline[f"{split}Indices"], dtype=np.int64)
               for split in ("training", "validation", "test")}
    if {split: len(part) for split, part in indices.items()} != cnn_report["partitionCounts"]:
        raise RuntimeError("Buffered partitions differ from the saved CNN experiment")
    if len(np.unique(np.concatenate(list(indices.values())))) != sum(map(len, indices.values())):
        raise RuntimeError("Buffered partitions overlap")
    for split, part in indices.items():
        if any(rows[int(index)]["partition"] != split for index in part):
            raise RuntimeError(f"{split} indices disagree with patch metadata")
    raw = np.asarray([features_by_id[row["location_id"]] for row in rows], dtype=np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    elevation = np.asarray([row["elevation_m"] for row in rows], dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    normalized = np.empty_like(raw)
    normalized[indices["training"]] = scaler.fit_transform(
        imputer.fit_transform(raw[indices["training"]]))
    for split in ("validation", "test"):
        normalized[indices[split]] = scaler.transform(imputer.transform(raw[indices[split]]))

    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "target": cnn_report["target"],
        "ablation": "CNN tabular branch and head, without any image-patch input",
        "pointFeatures": FEATURES,
        "seed": SEED,
        "epochs": EPOCHS,
        "batchSize": BATCH_SIZE,
        "optimizer": {"name": "AdamW", "learningRate": 0.001, "weightDecay": 0.01},
        "targetValidationRecall": TARGET_RECALL,
        "partitionCounts": cnn_report["partitionCounts"],
        "candidateHistory": [],
    }
    best = None
    for hard_minimal in (False, True):
        torch.manual_seed(SEED)
        model = PointOnlyHead()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        train = indices["training"]
        weights = np.where((labels[train] == 0) & (elevation[train] < 50) & hard_minimal,
                           1.5, 1.0).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(normalized[train]),
                                torch.from_numpy(labels[train].astype(np.float32)),
                                torch.from_numpy(weights))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        history = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            training_loss = 0.0
            for values, targets, sample_weights in loader:
                optimizer.zero_grad()
                loss = (nn.functional.binary_cross_entropy_with_logits(
                    model(values), targets, reduction="none") * sample_weights).mean()
                loss.backward()
                optimizer.step()
                training_loss += float(loss.item()) * len(targets)
            validation_scores = predict(model, normalized[indices["validation"]])
            threshold = threshold_for_recall(labels[indices["validation"]], validation_scores)
            evaluation = metrics(labels[indices["validation"]], validation_scores, threshold)
            candidate = {
                "hardMinimal": hard_minimal,
                "epoch": epoch,
                "trainingLoss": training_loss / len(dataset),
                "validationThreshold": threshold,
                "validation": evaluation,
            }
            history.append(candidate)
            rank = (evaluation["minimalFalsePositiveRate"], evaluation["brierScore"])
            if best is None or rank < best[0]:
                best = (rank, candidate, {key: value.clone() for key, value in model.state_dict().items()})
            if epoch == 1 or epoch % 10 == 0:
                print(f"weighted={hard_minimal} epoch={epoch} loss={candidate['trainingLoss']:.4f} "
                      f"val FPR={evaluation['minimalFalsePositiveRate']:.3f} "
                      f"recall={evaluation['onePercentRecall']:.3f}", flush=True)
        report["candidateHistory"].append({"hardMinimal": hard_minimal, "history": history})
    assert best is not None
    chosen = PointOnlyHead()
    chosen.load_state_dict(best[2])
    test_labels = labels[indices["test"]]
    point_scores = predict(chosen, normalized[indices["test"]])
    cnn_scores = np.load(CACHE / "selected-test-scores.npy")
    if len(cnn_scores) != len(test_labels):
        raise RuntimeError("Saved CNN test scores have a different row count")
    cnn_threshold = cnn_report["selected"]["validationThreshold"]
    cnn_check = metrics(test_labels, cnn_scores, cnn_threshold)
    saved_cnn_test = cnn_report["selected"]["test"]
    if (cnn_check["confusionMatrix"] != saved_cnn_test["confusionMatrix"] or
            not math.isclose(cnn_check["rocAuc"], saved_cnn_test["rocAuc"], abs_tol=1e-8) or
            not math.isclose(cnn_check["brierScore"], saved_cnn_test["brierScore"], abs_tol=1e-8)):
        raise RuntimeError("Saved CNN test scores do not match its report")
    selected = best[1]
    point_test = metrics(test_labels, point_scores, selected["validationThreshold"])
    cnn_test = cnn_report["selected"]["test"]
    detections = min(cnn_test["confusionMatrix"][1][1], point_test["confusionMatrix"][1][1])
    report["selected"] = {
        "hardMinimal": selected["hardMinimal"],
        "epoch": selected["epoch"],
        "validationThreshold": selected["validationThreshold"],
        "validation": selected["validation"],
        "test": point_test,
    }
    report["reference"] = {
        "cnn": cnn_test,
        "bufferedBoosted": cnn_report["bufferedTabularBaseline"]["test"],
    }
    report["matchedTestRecallDiagnostic"] = {
        "meaning": "Thresholds chosen from test labels for ranking comparison only; not deployable",
        "onePercentDetections": detections,
        "pointOnlyMinimalFalsePositives": matched_recall_errors(test_labels, point_scores, detections),
        "cnnMinimalFalsePositives": matched_recall_errors(test_labels, cnn_scores, detections),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows_md = [
        "# Terrain CNN point-only ablation", "",
        "The point-only network keeps the CNN's 10-to-32 tabular branch and 64-unit output head "
        "but removes its 64 × 64 image encoder. Both candidates use the same buffered rows, "
        "training-only median imputation and scaling, optimizer, validation recall rule, "
        "and low-elevation weighting options as the original CNN experiment. The point-only "
        "model is allowed up to 100 epochs because its validation error was still falling at "
        "the CNN's 14-epoch limit. Its longer selection window favors the point-only model.", "",
        f"Buffered rows: {len(indices['training'])} training, {len(indices['validation'])} validation, "
        f"{len(indices['test'])} test. Selected weighted={selected['hardMinimal']} "
        f"at epoch {selected['epoch']} using validation only.", "",
        "| Test metric | Point only | CNN + point inputs | Buffered boosted |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key, fmt in [
        ("Minimal false positives", "minimalFalsePositives", ".0f"),
        ("Minimal false-positive rate", "minimalFalsePositiveRate", ".3f"),
        ("1% recall", "onePercentRecall", ".3f"),
        ("Balanced accuracy", "balancedAccuracy", ".3f"),
        ("ROC AUC", "rocAuc", ".3f"),
        ("Brier score", "brierScore", ".3f"),
    ]:
        values = (point_test[key], cnn_test[key], report["reference"]["bufferedBoosted"][key])
        rows_md.append(f"| {label} | " + " | ".join(format(value, fmt) for value in values) + " |")
    diagnostic = report["matchedTestRecallDiagnostic"]
    rows_md += [
        "",
        f"At the same post-hoc test recall of {detections}/{int((test_labels == 1).sum())}, "
        f"point only has {diagnostic['pointOnlyMinimalFalsePositives']} and the CNN has "
        f"{diagnostic['cnnMinimalFalsePositives']} Minimal false positives. "
        "This test-label threshold is a ranking diagnostic, not a deployable cutoff.", "",
        "The image-enabled CNN performs better in this run, including at matched recall. "
        "This comparison alone does not isolate the spatial pixels: removing the image encoder "
        "also changes model capacity. A patch-shuffled CNN with the same architecture would "
        "provide a tighter control.", "",
        "The test partition has been used in earlier model studies and is not a fresh independent "
        "holdout. These labels are FEMA map classes, not observed storm inundation.", "",
        "Reproduce with `npm run model:experiment:terrain-cnn-point-only`.", "",
    ]
    REPORT_MD.write_text("\n".join(rows_md), encoding="utf-8")
    print(f"Selected weighted={selected['hardMinimal']} epoch={selected['epoch']}; "
          f"test point-only FPR={point_test['minimalFalsePositiveRate']:.3f}, "
          f"CNN FPR={cnn_test['minimalFalsePositiveRate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
