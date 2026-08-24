# Gradient-boosted physical hazard benchmark

The untouched test set contains geographic blocks that were not used for training or hyperparameter selection.
Hyperparameters were selected by the lowest validation Brier score because the app outputs probabilities; AUC breaks ties.

| Metric | Current logistic | Gradient boosted | Change |
| --- | ---: | ---: | ---: |
| Class accuracy | 83.538% | 84.923% | +1.385% |
| Balanced accuracy | 61.788% | 62.319% | +0.531% |
| Macro F1 | 0.5852 | 0.6082 | +0.0230 |
| 1% AUC | 0.9825 | 0.9875 | +0.0050 |
| >=0.2% AUC | 0.8830 | 0.9016 | +0.0186 |
| 1% Brier (lower is better) | 0.0473 | 0.0336 | -0.0137 |
| >=0.2% Brier (lower is better) | 0.1114 | 0.1050 | -0.0064 |

## Class recall

| FEMA-equivalent class | Current logistic | Gradient boosted |
| --- | ---: | ---: |
| one_percent | 89.181% | 87.135% |
| point_two_percent | 0.000% | 0.685% |
| minimal | 96.182% | 99.138% |

Recommendation: **use the gradient-boosted production model with the parity test retained**

Important limitation: the middle 0.2% class remains difficult to identify. These labels are FEMA-equivalent hazard classes, not observed 100-year flood outcomes at individual properties.

The gradient-boosted trees are exported to the production model artifact used by the app.
