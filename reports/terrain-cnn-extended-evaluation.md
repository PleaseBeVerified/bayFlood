# Point-centered terrain CNN FEMA map-class experiment — extended training

This offline model predicts FEMA `one_percent` versus `minimal` map labels, not observed flooding. Each GPS coordinate centers a 64 × 64 grid of 30 m raster pixels. A 1,920 m Chebyshev buffer removes training locations whose patches could overlap a validation or test patch; validation patches near test patches are also removed.

Buffered rows: training 4998, validation 1067, test 1154. Model selection and the 95% recall threshold use validation only.

Training: up to 80 epochs, AdamW learning rate 0.0003, early-stopping patience 15; selected weighted epoch 5.

| Test metric | Buffered terrain XGBoost | CNN + point features |
| --- | ---: | ---: |
| Minimal false positives | 132 | 67 |
| Minimal false-positive rate | 0.163 | 0.083 |
| 1% recall | 0.988 | 0.947 |
| Balanced accuracy | 0.913 | 0.932 |
| ROC AUC | 0.992 | 0.985 |
| Brier score | 0.031 | 0.050 |

Minimal-class false positives by elevation:

| Elevation | Rows | XGBoost errors | CNN errors |
| --- | ---: | ---: | ---: |
| below 10m | 51 | 37 | 28 |
| 10 to 50m | 146 | 49 | 24 |
| 50 to 282m | 431 | 42 | 14 |
| at least 282m | 184 | 4 | 1 |

At the same **test** recall (324/342), a post-hoc ranking check gives 41 XGBoost versus 63 CNN `minimal` false positives. These test-derived thresholds are diagnostic only and cannot be used for deployment.

The same geographic test partition has been used for earlier model studies, so it is a diagnostic comparison rather than fresh independent confirmation. Neither model was evaluated against observed inundation at FEMA-minimal locations.
