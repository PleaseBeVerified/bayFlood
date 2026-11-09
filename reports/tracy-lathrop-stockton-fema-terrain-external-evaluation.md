# Tracy–Lathrop–Stockton external model test

This uses 450 western San Joaquin County points held out from existing model training:
150 each from FEMA NFHL mapped minimal, 0.2%, and 1% annual-chance classes.
The labels describe mapped hazard, not observed event flooding. No model was retrained.

The sample and nearest-water feature use 9,200 local USGS 3DHP
flowlines plus existing local hydrography from the outset. The study window
is (-121.55, 37.68, -121.15, 37.98), limited to San Joaquin FEMA map 06077C.

| Existing model | Balanced accuracy | Macro F1 | 0.2% class F1 |
| --- | ---: | ---: | ---: |
| Terrain boosted | 0.518 | 0.416 | 0.013 |
| Terrain DNN | 0.471 | 0.416 | 0.221 |
| Five-input DNN | 0.547 | 0.493 | 0.192 |

Confusion matrices use rows = FEMA class and columns = predicted class,
in the order minimal, 1%, 0.2%:

- Terrain boosted: `[[94, 56, 0], [12, 138, 0], [54, 95, 1]]`
- Terrain DNN: `[[45, 90, 15], [0, 146, 4], [2, 127, 21]]`
- Five-input DNN: `[[83, 61, 6], [0, 146, 4], [13, 120, 17]]`

The existing binary models are scored on the 300 explicit minimal and 1%
points. The 0.2% points have no binary target and are excluded:

| Binary terrain model | Balanced accuracy | 1% F1 | ROC AUC |
| --- | ---: | ---: | ---: |
| Served terrain boosted | 0.607 | 0.718 | 0.922 |
| Terrain DNN | 0.647 | 0.739 | 0.891 |
| High-elevation terrain boosted | 0.500 | 0.667 | 0.841 |
| Five-input boosted | 0.693 | 0.765 | 0.929 |
| Five-input DNN | 0.660 | 0.746 | 0.922 |
| Terrain CNN | 0.653 | 0.743 | 0.773 |

The binary thresholds come from the original validation region;
they were not tuned on San Joaquin. Confusion matrices use rows = FEMA class
and columns = predicted class, in the order minimal, 1%:

- Served terrain boosted: `[[32, 118], [0, 150]]`
- Terrain DNN: `[[44, 106], [0, 150]]`
- High-elevation terrain boosted: `[[0, 150], [0, 150]]`
- Five-input boosted: `[[58, 92], [0, 150]]`
- Five-input DNN: `[[48, 102], [0, 150]]`
- Terrain CNN: `[[46, 104], [0, 150]]`

The high-elevation training variant detects 150 of 150 mapped 1% points and flags 150 of 150 mapped minimal points.

The CNN uses 64×64 patches at 30 m resolution. Its Bay Area test
balanced accuracy was 0.938 and 1% recall was 0.959 with the same saved checkpoint.
Of the 300 San Joaquin patches, 0 have less than 90% valid topography; the CNN receives validity masks.

## Pattern among the CNN's mapped-minimal false positives

The CNN detects all 150 mapped 1% points, but flags 104 of 150 mapped minimal points. Medians for flagged
versus correctly cleared minimal points:

| Terrain input | CNN flagged minimal | CNN cleared minimal |
| --- | ---: | ---: |
| Height above drainage | 1.2 m | 7.5 m |
| Nearest mapped water | 324.0 m | 427.0 m |
| Elevation relative to 250 m neighborhood | 0.0 m | -0.0 m |
| Elevation relative to 500 m neighborhood | 0.0 m | -0.1 m |
| Flow accumulation | 1,500.0 m² | 2,900.0 m² |
| Elevation | 13.0 m | 38.0 m |

These group medians are descriptive; the features and FEMA polygons are
spatially correlated. Polygon 28049400 alone contributes 58 of the CNN false positives. Every CNN patch has at least 90%
valid topography in this sample.

This balanced, spatially sampled set is an external transfer check.
It does not estimate real-world hazard prevalence or event prediction accuracy.
