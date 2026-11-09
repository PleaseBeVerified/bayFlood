# Northern Santa Cruz external terrain model test

This uses 450 Santa Cruz County points held out from existing model training:
150 each from FEMA NFHL mapped minimal, 0.2%, and 1% annual-chance classes.
The labels describe mapped hazard, not observed event flooding. No model was retrained.

The local water layer initially missed Santa Cruz river flowlines. This evaluation
includes 5,111 USGS 3DHP flowlines; all 450 point locations and labels
were kept fixed while nearest-water distances were corrected. These results
supersede the first Santa Cruz scores. Original point eligibility still used
the older water layer, so the sample is not a random regional draw.

| Existing model | Balanced accuracy | Macro F1 | 0.2% class F1 |
| --- | ---: | ---: | ---: |
| Terrain boosted | 0.493 | 0.438 | 0.198 |
| Terrain DNN | 0.456 | 0.384 | 0.150 |
| Five-input DNN | 0.458 | 0.369 | 0.039 |

Confusion matrices use rows = FEMA class and columns = predicted class,
in the order minimal, 1%, 0.2%:

- Terrain boosted: `[[145, 1, 4], [81, 59, 10], [121, 11, 18]]`
- Terrain DNN: `[[148, 0, 2], [98, 44, 8], [130, 7, 13]]`
- Five-input DNN: `[[150, 0, 0], [97, 53, 0], [143, 4, 3]]`

The existing binary models are scored on the 300 explicit minimal and 1%
points. The 0.2% points have no binary target and are excluded:

| Binary terrain model | Balanced accuracy | 1% F1 | ROC AUC |
| --- | ---: | ---: | ---: |
| Served terrain boosted | 0.727 | 0.640 | 0.977 |
| Terrain DNN | 0.777 | 0.720 | 0.976 |
| High-elevation terrain boosted | 0.787 | 0.824 | 0.996 |
| Five-input boosted | 0.587 | 0.295 | 0.927 |
| Five-input DNN | 0.687 | 0.552 | 0.960 |
| Terrain CNN | 0.837 | 0.805 | 0.997 |

The binary thresholds come from the original validation region;
they were not tuned on Santa Cruz. Confusion matrices use rows = FEMA class
and columns = predicted class, in the order minimal, 1%:

- Served terrain boosted: `[[145, 5], [77, 73]]`
- Terrain DNN: `[[147, 3], [64, 86]]`
- High-elevation terrain boosted: `[[86, 64], [0, 150]]`
- Five-input boosted: `[[150, 0], [124, 26]]`
- Five-input DNN: `[[148, 2], [92, 58]]`
- Terrain CNN: `[[150, 0], [49, 101]]`

The high-elevation training variant detects every mapped 1% point in
this holdout, but also flags 64 of 150 mapped minimal points.

The CNN uses 64×64 patches at 30 m resolution. Its Bay Area test
balanced accuracy was 0.938 and 1% recall was 0.959 with the same saved checkpoint.
Of the 300 Santa Cruz patches, 32 have less than 90% valid topography; the CNN receives validity masks.

## Pattern among missed mapped 1% points

The CNN misses 49 of 150 mapped 1% points; 35 are also missed by the served boosted model.
Medians for the CNN missed versus detected 1% points:

| Terrain input | CNN missed | CNN detected |
| --- | ---: | ---: |
| Height above drainage | 7.7 m | 4.5 m |
| Nearest mapped water | 75.0 m | 33.0 m |
| Elevation relative to 250 m neighborhood | -1.9 m | -19.1 m |
| Elevation relative to 500 m neighborhood | -9.5 m | -45.5 m |
| Flow accumulation | 2,100.0 m² | 8,100.0 m² |
| Elevation | 79.1 m | 68.2 m |

Missed points tend to have higher height above drainage, sit farther
from mapped water, and have weaker valley/depression signals. This is
descriptive, not a causal attribution: the features and FEMA polygons are
spatially correlated. Polygon 352456 alone contributes 18 of the CNN misses. Only 3 of the 49 CNN misses have less than 90% valid topography in their patches.

This balanced, spatially sampled set is an external transfer check.
It does not estimate real-world hazard prevalence or event prediction accuracy.
