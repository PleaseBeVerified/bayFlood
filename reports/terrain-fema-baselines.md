# Terrain-augmented FEMA map-class baselines

These models add five terrain features to the current five-input FEMA classifier. They predict FEMA map classes, not flooding during a specified storm. Both variants use the same unchanged geographic validation and test splits. The high-elevation variant matches the number of positive training rows with the highest-elevation minimal rows; its cutoff is 282.1 m.

| Geographic test metric | Previous app | Terrain, full training | Terrain, high-elevation sample |
| --- | ---: | ---: | ---: |
| Accuracy | 0.938 | 0.915 | 0.453 |
| Balanced accuracy | 0.944 | 0.935 | 0.610 |
| Precision | 0.854 | 0.784 | 0.351 |
| Recall | 0.956 | 0.985 | 0.994 |
| F1 | 0.902 | 0.873 | 0.519 |
| ROC AUC | 0.987 | 0.994 | 0.948 |
| Brier score | 0.032 | 0.026 | 0.530 |

The full-training terrain model is exported for the app. These test results cannot validate performance for locations without a FEMA map class or for observed storm events.
