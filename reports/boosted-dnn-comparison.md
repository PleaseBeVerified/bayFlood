# Boosted model comparison

This model uses the same five inputs and geographic partitions as the DNN. Hyperparameters and the decision threshold were chosen using training and validation data. The test split evaluates agreement with FEMA-derived mapped classes, not actual property floods.

- Validation balanced accuracy: 0.894
- Test accuracy: 0.938
- Test balanced accuracy: 0.944
- Test precision: 0.854
- Test recall: 0.956
- Test F1: 0.902
- Test ROC AUC: 0.987
- Confusion matrix [[TN, FP], [FN, TP]]: [[756, 56], [15, 327]]

| Geographic test metric | Earlier DNN | Boosted app model |
| --- | ---: | ---: |
| Accuracy | 0.901 | 0.938 |
| Balanced accuracy | 0.920 | 0.944 |
| Recall | 0.968 | 0.956 |
| F1 | 0.853 | 0.902 |

The boosted model reduces false positives from 103 to 56 on this split, while false negatives rise from 11 to 15. An independent future geographic holdout is needed to confirm the gain after this model comparison.

The DNN remains available for research comparison. The app uses this boosted model for its separate current-condition pattern score. FEMA map classes still determine the displayed flood-event benchmarks.
