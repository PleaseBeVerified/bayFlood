# DNN Model Evaluation

This report evaluates a tabular feedforward neural network on spatially separated
Bay Area blocks. Architecture selection and threshold tuning used validation data;
the test partition remained untouched until final evaluation.

## Data

- Training rows: 6690
- Validation rows: 1223
- Test rows: 1154
- Features: elevation_m, current_imperviousness_pct, current_100yr_24h_rainfall_mm, nearest_water_available, log_nearest_water_distance_m
- Hidden layers: [32, 16]
- Decision threshold: 0.665

## Untouched Test Results

- Accuracy: 0.9012
- Balanced accuracy: 0.9205
- ROC AUC: 0.9811
- Average precision: 0.9639
- Precision: 0.7627
- Recall: 0.9678
- F1: 0.8531
- Brier score: 0.1032
- Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[709, 103], [11, 331]]`

Accuracy alone is not sufficient because the dataset is imbalanced. Balanced
accuracy, ROC AUC, precision, recall, and calibration should be considered together.
The FEMA-derived target is a screening label, not a literal probability that a
specific property will flood.
