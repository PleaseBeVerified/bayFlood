# Three-class CNN external geographic evaluation

The saved CNN and its buffered boosted comparator were scored without
retraining on 450 northern Santa Cruz FEMA map-class points, 150 per class.
This region is absent from the Bay Area training table.

This report uses the corrected water-distance inputs with 5,111
USGS 3DHP flowlines. The 450 point locations and labels are unchanged;
the earlier three-class CNN Santa Cruz scores are superseded.

| Metric | CNN | Buffered boosted |
| --- | ---: | ---: |
| Accuracy | 0.436 | 0.533 |
| Balanced accuracy | 0.436 | 0.533 |
| Macro F1 | 0.339 | 0.505 |
| Log loss | 1.125 | 1.059 |

| Class | CNN F1 | Boosted F1 |
| --- | ---: | ---: |
| minimal | 0.568 | 0.609 |
| one_percent | 0.387 | 0.486 |
| point_two_percent | 0.063 | 0.418 |

CNN confusion matrix: `[[149, 0, 1], [106, 42, 2], [120, 25, 5]]`.
Boosted confusion matrix: `[[142, 2, 6], [80, 52, 18], [94, 10, 46]]`.

The sample is intentionally balanced and its points can be spatially
correlated within narrow polygons. The labels are FEMA map categories,
not observed floods or property flood probabilities.
