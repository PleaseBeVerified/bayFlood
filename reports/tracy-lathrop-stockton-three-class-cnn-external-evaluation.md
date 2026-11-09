# Tracy–Lathrop–Stockton three-class CNN external evaluation

The saved CNN and its buffered boosted comparator were scored without
retraining on 450 western San Joaquin FEMA map-class points, 150 per class.
This region is absent from the Bay Area training table.

The sample and water-distance inputs use 9,200 local
USGS 3DHP flowlines from the outset.

| Metric | CNN | Buffered boosted |
| --- | ---: | ---: |
| Accuracy | 0.447 | 0.538 |
| Balanced accuracy | 0.447 | 0.538 |
| Macro F1 | 0.382 | 0.430 |
| Log loss | 1.304 | 1.094 |

| Class | CNN F1 | Boosted F1 |
| --- | ---: | ---: |
| minimal | 0.295 | 0.615 |
| one_percent | 0.599 | 0.675 |
| point_two_percent | 0.250 | 0.000 |

CNN confusion matrix: `[[26, 69, 55], [0, 145, 5], [0, 120, 30]]`.
Boosted confusion matrix: `[[114, 36, 0], [22, 128, 0], [85, 65, 0]]`.

The sample is intentionally balanced and its points can be spatially
correlated within narrow polygons. The labels are FEMA map categories,
not observed floods or property flood probabilities.
