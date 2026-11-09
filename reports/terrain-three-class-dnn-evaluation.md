# Three-class terrain DNN evaluation

All 10,000 FEMA-labeled samples are used, including 933 mapped 0.2% annual-chance locations. The model predicts mapped minimal, 0.2%, or 1% hazard categories from the same ten features as the terrain binary models. Eight candidates were selected by geographic-validation macro F1. This three-class model remains available for offline research; the app offers binary boosted, DNN, and CNN scores.

| Geographic test metric | Five-input three-class DNN | Ten-input three-class DNN |
| --- | ---: | ---: |
| Accuracy | 0.778 | 0.789 |
| Balanced accuracy | 0.787 | 0.740 |
| Macro F1 | 0.723 | 0.709 |
| Log loss | 0.542 | 0.614 |

| Ten-input test class | Samples | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| minimal | 812 | 0.911 | 0.793 | 0.848 |
| one_percent | 342 | 0.802 | 0.886 | 0.842 |
| point_two_percent | 146 | 0.367 | 0.541 | 0.438 |

Confusion matrix (rows observed, columns predicted; class order minimal, one_percent, point_two_percent): `[[644, 61, 107], [10, 303, 29], [53, 14, 79]]`.

These are FEMA map-class labels, not observed floods. An unmapped address has no independent label in this evaluation, and the class scores are not annual flooding probabilities.
