# High-elevation FEMA-label sampling experiment

Selected all 1238 training rows mapped to the 1% class and the 1238 highest-elevation training rows mapped minimal. The selected minimal cutoff is 282.1 m. Validation and test geographic partitions retain their original class mix. The threshold was selected on validation only.

| Geographic test metric | Current app model | High-elevation sample |
| --- | ---: | ---: |
| Accuracy | 0.938 | 0.449 |
| Balanced accuracy | 0.944 | 0.603 |
| Precision | 0.854 | 0.348 |
| Recall | 0.956 | 0.982 |
| F1 | 0.902 | 0.514 |
| Brier score | 0.032 | 0.544 |

On test minimal rows below 282.1 m, the false-positive rate is 1.000 (628 rows). At or above the cutoff, it is 0.011 (184 rows).

These labels are FEMA mapped classes, not verified storm flooding. This experiment must not be presented as event-conditioned flood probability.
