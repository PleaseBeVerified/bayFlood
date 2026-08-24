# Balanced three-class XGBoost benchmark

The production model was not changed. Hyperparameters, probability calibration, and final evaluation use separate geographic blocks.

| Test metric | Ordered production | Three-class uncalibrated | Three-class calibrated |
| --- | ---: | ---: | ---: |
| Accuracy | 84.9% | 83.9% | 86.2% |
| Balanced accuracy | 62.3% | 77.5% | 68.1% |
| Macro F1 | 0.6082 | 0.7584 | 0.6883 |
| 0.2% recall | 0.7% | 53.4% | 15.1% |
| 0.2% precision | 16.7% | 45.9% | 57.9% |
| Log loss (lower is better) | 0.4162 | 0.3745 | 0.3349 |
| Multiclass Brier (lower is better) | 0.2255 | 0.2191 | 0.1914 |
| Calibration error (lower is better) | 0.0209 | 0.0228 | 0.0277 |

## Split

| Purpose | Rows | Geographic blocks | 0.2% rows |
| --- | ---: | ---: | ---: |
| Training | 7364 | 463 | 674 |
| Tuning | 667 | 42 | 51 |
| Calibration | 669 | 33 | 62 |
| Test | 1300 | 79 | 146 |

The direct model is trained with inverse-frequency class weights. Vector scaling is then fitted on the calibration-only blocks to correct the weighted model's probabilities.

## Conclusion

The calibrated model changes 0.2% recall from 0.7% to 15.1%, while changing overall accuracy from 84.9% to 86.2%. Log loss and multiclass Brier score also improve. Expected calibration error is slightly worse on the test set, so it remains explicitly reported rather than treated as a universal calibration win.

Recommendation: advance this benchmark to physical-behavior and stability audits before considering production integration. Direct multiclass trees do not currently carry the ordered model's monotonic guarantees.

Important: these remain FEMA-equivalent hazard-class probabilities, not observed parcel flood outcomes.
