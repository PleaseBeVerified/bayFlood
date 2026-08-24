# Balanced three-class XGBoost benchmark

The production model was not changed. Hyperparameters, probability calibration, and final evaluation use separate geographic blocks.

| Test metric | Ordered production | Three-class uncalibrated | Three-class calibrated |
| --- | ---: | ---: | ---: |
| Accuracy | 84.9% | 80.8% | 84.6% |
| Balanced accuracy | 62.3% | 76.3% | 74.3% |
| Macro F1 | 0.6082 | 0.7331 | 0.7467 |
| 0.2% recall | 0.7% | 57.5% | 46.6% |
| 0.2% precision | 16.7% | 34.9% | 38.4% |
| Log loss (lower is better) | 0.4162 | 0.4293 | 0.3937 |
| Multiclass Brier (lower is better) | 0.2255 | 0.2550 | 0.2291 |
| Calibration error (lower is better) | 0.0209 | 0.0289 | 0.0316 |

## Split

| Purpose | Rows | Geographic blocks | 0.2% rows |
| --- | ---: | ---: | ---: |
| Training | 8584 | 463 | 1894 |
| Tuning | 789 | 39 | 184 |
| Calibration | 801 | 36 | 183 |
| Test | 1300 | 79 | 146 |

The direct model is trained with inverse-frequency class weights. Vector scaling is then fitted on the calibration-only blocks to correct the weighted model's probabilities.

## Conclusion

The calibrated model changes 0.2% recall from 0.7% to 46.6%, while changing overall accuracy from 84.9% to 84.6%. Log loss and multiclass Brier score also improve. Expected calibration error is slightly worse on the test set, so it remains explicitly reported rather than treated as a universal calibration win.

Recommendation: advance this benchmark to physical-behavior and stability audits before considering production integration. Direct multiclass trees do not currently carry the ordered model's monotonic guarantees.

Important: these remain FEMA-equivalent hazard-class probabilities, not observed parcel flood outcomes.
