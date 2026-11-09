# Terrain CNN point-only ablation

The point-only network keeps the CNN's 10-to-32 tabular branch and 64-unit output head but removes its 64 × 64 image encoder. Both candidates use the same buffered rows, training-only median imputation and scaling, optimizer, validation recall rule, and low-elevation weighting options as the original CNN experiment. The point-only model is allowed up to 100 epochs because its validation error was still falling at the CNN's 14-epoch limit. Its longer selection window favors the point-only model.

Buffered rows: 4998 training, 1067 validation, 1154 test. Selected weighted=False at epoch 59 using validation only.

| Test metric | Point only | CNN + point inputs | Buffered boosted |
| --- | ---: | ---: | ---: |
| Minimal false positives | 115 | 68 | 132 |
| Minimal false-positive rate | 0.142 | 0.084 | 0.163 |
| 1% recall | 0.974 | 0.959 | 0.988 |
| Balanced accuracy | 0.916 | 0.938 | 0.913 |
| ROC AUC | 0.986 | 0.988 | 0.992 |
| Brier score | 0.046 | 0.045 | 0.031 |

At the same post-hoc test recall of 328/342, point only has 73 and the CNN has 55 Minimal false positives. This test-label threshold is a ranking diagnostic, not a deployable cutoff.

The image-enabled CNN performs better in this run, including at matched recall. This comparison alone does not isolate the spatial pixels: removing the image encoder also changes model capacity. A patch-shuffled CNN with the same architecture would provide a tighter control.

The test partition has been used in earlier model studies and is not a fresh independent holdout. These labels are FEMA map classes, not observed storm inundation.

Reproduce with `npm run model:experiment:terrain-cnn-point-only`.
