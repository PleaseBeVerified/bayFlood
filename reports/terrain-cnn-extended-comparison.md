# Longer, lower-rate terrain CNN training

The extended run uses the same 4,998 training, 1,067 validation, and 1,154 test
locations as the saved CNN. It lowers AdamW's learning rate from 0.001 to
0.0003, allows up to 80 epochs instead of 14, and stops each weighting candidate
after 15 epochs without an improvement in validation `minimal` false-positive
rate at at least 95% `one_percent` recall (Brier score breaks ties). It writes a
separate checkpoint; the app still uses the original model.

| Metric | Current CNN | Extended run |
| --- | ---: | ---: |
| Validation `minimal` false positives | 89/862 | 83/862 |
| Validation `one_percent` recall | 195/205 | 195/205 |
| Selected candidate and epoch | Unweighted, 10 | Weighted, 5 |
| Bay Area test `minimal` false positives | 68/812 | 67/812 |
| Bay Area test `one_percent` detected | 328/342 | 324/342 |
| Bay Area test ROC AUC | 0.988 | 0.985 |
| Bay Area test Brier score | 0.045 | 0.050 |

At the same **post-hoc test recall** of 324/342, the current model produces 49
`minimal` false positives and the extended model 63. At 328/342, the counts are
55 and 94. These test-label thresholds are ranking diagnostics, not usable
deployment thresholds.

On the separate northern Santa Cruz sample, both checkpoints classify 101/150
`one_percent` points and 0/150 `minimal` points as higher-risk at their respective
validation thresholds. Their ROC AUC values are 0.9971 (current) and 0.9968
(extended).

The extended training improves the selected validation false-positive count,
but does not improve the Bay Area test tradeoff or Santa Cruz classifications.
It is not promoted to the app. Neither test set is a fresh holdout for this
project, and the labels represent FEMA map classes rather than observed floods.

Run `npm run model:experiment:terrain-cnn-extended` to reproduce the training.
The full candidate histories and test metrics are in
[`terrain-cnn-extended-evaluation.json`](terrain-cnn-extended-evaluation.json).
