# Three-class data amount and class-weight search

All data-fraction, weight-strength, and tree candidates were selected on original tuning blocks. Added validation rows were excluded. Vector scaling was fitted once on original calibration blocks, and the untouched test set was evaluated once after selection.

## Selected configuration

- Added training data: 75% (915 rows)
- Class-weight strength: 0.35 (0 is none; 1 is fully balanced)
- Trees: 1647
- Hyperparameters: `{"eta": 0.03, "max_depth": 4, "min_child_weight": 20, "reg_lambda": 10}`

## Untouched geographic test

| Metric | Ordered production | Search winner |
| --- | ---: | ---: |
| Accuracy | 84.92% | 85.46% |
| Balanced accuracy | 62.32% | 64.35% |
| Macro F1 | 0.6082 | 0.6227 |
| 0.2% recall | 0.68% | 2.74% |
| 0.2% precision | 16.67% | 36.36% |
| Log loss | 0.4162 | 0.3621 |
| Multiclass Brier | 0.2255 | 0.2063 |
| Calibration error | 0.0209 | 0.0265 |

The earlier calibrated three-class benchmark had 86.23% accuracy and 15.07% 0.2% recall. It was not used to select this search winner.

The production app model was not changed. This remains a FEMA-equivalent physical-hazard benchmark, not a parcel flood-outcome model.
