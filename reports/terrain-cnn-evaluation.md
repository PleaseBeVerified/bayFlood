# Point-centered terrain CNN FEMA map-class experiment

This offline model predicts FEMA `one_percent` versus `minimal` map labels, not observed flooding. Each GPS coordinate centers a 64 × 64 grid of 30 m raster pixels. A 1,920 m Chebyshev buffer removes training locations whose patches could overlap a validation or test patch; validation patches near test patches are also removed.

Buffered rows: training 4998, validation 1067, test 1154. Model selection and the 95% recall threshold use validation only.

| Test metric | Buffered terrain XGBoost | CNN + point features |
| --- | ---: | ---: |
| Minimal false positives | 132 | 68 |
| Minimal false-positive rate | 0.163 | 0.084 |
| 1% recall | 0.988 | 0.959 |
| Balanced accuracy | 0.913 | 0.938 |
| ROC AUC | 0.992 | 0.988 |
| Brier score | 0.031 | 0.045 |

Minimal-class false positives by elevation:

| Elevation | Rows | XGBoost errors | CNN errors |
| --- | ---: | ---: | ---: |
| below 10m | 51 | 37 | 23 |
| 10 to 50m | 146 | 49 | 23 |
| 50 to 282m | 431 | 42 | 20 |
| at least 282m | 184 | 4 | 2 |

At the same **test** recall (328/342), a post-hoc ranking check gives 54 XGBoost versus 55 CNN `minimal` false positives. These test-derived thresholds are diagnostic only. The validation-selected CNN threshold reduces false positives partly by accepting lower test recall; its Brier score is also worse.

The same geographic test partition has been used for earlier model studies, so it is a diagnostic comparison rather than fresh independent confirmation. Neither model was evaluated against observed inundation at FEMA-minimal locations.
