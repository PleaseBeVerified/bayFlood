# Ten-input terrain DNN evaluation

The DNN uses the same ten inputs, FEMA map-class labels, and geographic splits as the terrain boosted model. Eighteen architecture and optimization candidates were ranked by geographic-validation ROC AUC; the decision threshold was selected on validation data. The selected network has 128, 64, and 32 hidden units, alpha 0.01, and no class weights. The app offers this DNN alongside the boosted model.

| Geographic test metric | Five-input DNN | Initial terrain DNN | Tuned terrain DNN | Terrain boosted |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.901 | 0.899 | 0.910 | 0.915 |
| Balanced accuracy | 0.920 | 0.918 | 0.928 | 0.935 |
| Precision | 0.763 | 0.759 | 0.778 | 0.784 |
| Recall | 0.968 | 0.965 | 0.974 | 0.985 |
| F1 | 0.853 | 0.849 | 0.865 | 0.873 |
| ROC AUC | 0.981 | 0.988 | 0.989 | 0.994 |
| Brier score | 0.103 | 0.059 | 0.037 | 0.026 |

These results measure FEMA map-class agreement. They do not measure the chance that a property floods during a specified storm, nor performance on addresses without a FEMA map class. The same test split has been used for earlier model comparisons, so these numbers are not independent confirmation on a fresh holdout.
