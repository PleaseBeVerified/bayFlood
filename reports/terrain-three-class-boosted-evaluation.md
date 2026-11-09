# Three-class terrain boosted evaluation

The model uses all 10,000 FEMA-labeled samples, including 933 mapped 0.2% annual-chance locations, with the ten terrain inputs. Eight candidates were ranked by geographic-validation macro F1. The app continues to serve the binary terrain boosted score.

| Geographic test metric | Five-input DNN | Terrain DNN | Terrain boosted |
| --- | ---: | ---: | ---: |
| Accuracy | 0.778 | 0.789 | 0.841 |
| Balanced accuracy | 0.787 | 0.740 | 0.788 |
| Macro F1 | 0.723 | 0.709 | 0.768 |
| Log loss | 0.542 | 0.614 | 0.374 |

| Terrain boosted test class | Samples | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| minimal | 812 | 0.915 | 0.847 | 0.880 |
| one_percent | 342 | 0.860 | 0.936 | 0.896 |
| point_two_percent | 146 | 0.483 | 0.582 | 0.528 |

Confusion matrix (rows observed, columns predicted; class order minimal, one_percent, point_two_percent): `[[688, 44, 80], [11, 320, 11], [53, 8, 85]]`.

The 0.2% class remains difficult to distinguish. These are FEMA map classes, not observed flood events; an unmapped address has no independent label in this evaluation.
