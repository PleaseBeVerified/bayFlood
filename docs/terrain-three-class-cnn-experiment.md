# Three-class terrain CNN experiment

The three-class CNN uses the 933 FEMA-mapped 0.2%-annual-chance samples as a
separate target alongside `minimal` and `one_percent`. It is an offline
map-class experiment; the app's displayed percentages still come from FEMA
polygons. A predicted class is neither an observed flood outcome nor an annual
flood probability.

## Reproduce

With the source rasters and `.venv-model` dependencies described in
[the binary CNN experiment](terrain-cnn-experiment.md):

```sh
npm run model:cnn:three-class-patches
npm run model:experiment:terrain-three-class-cnn
npm run model:cnn:three-class-external-patches
npm run model:experiment:terrain-three-class-cnn-external
```

The three-class patch cache is separate from the existing binary CNN cache.
The 64 × 64 patches have the same nine channels and 30 m resolution as the
binary CNN. FEMA zone polygons are not image channels. The network combines
patch features with the same ten point features as the terrain boosted model.
The saved model is `models/flood-hazard-terrain-three-class-cnn.pt`.

The original geographic partitions hold 7,364 training, 1,336 validation,
and 1,300 test points. A 1,920 m buffer removes training and validation
points whose image patches could overlap held-out patches. The resulting
partitions contain 5,521 training, 1,166 validation, and 1,300 test points;
the 0.2% counts are 539, 100, and 146 respectively. The CNN runs with ordinary
and balanced class weights for 12 epochs each. The checkpoint is selected by
validation macro F1, then 0.2% F1 and log loss. The boosted comparator is
trained and selected on those **same buffered points**, with the same ten
point features and no patch pixels. The test partition is scored only after
model selection.

## Results

| Geographic test set | Model | Macro F1 | 0.2% F1 | 0.2% recall |
| --- | --- | ---: | ---: | ---: |
| Bay Area, 1,300 points | Three-class CNN | 0.742 | 0.476 | 0.582 |
| Bay Area, same points | Buffered terrain boosted | 0.749 | 0.481 | 0.514 |
| Northern Santa Cruz, 450 balanced points | Three-class CNN | 0.339 | 0.063 | 0.033 |
| Northern Santa Cruz, same points | Buffered terrain boosted | 0.505 | 0.418 | 0.307 |

The CNN's higher 0.2% recall on the Bay Area test comes with lower precision
(0.403 versus 0.452), leaving its 0.2% F1 and overall macro F1 slightly
below the boosted comparator. On the Santa Cruz sample the CNN recognizes
only 5 of 150 mapped 0.2% points, compared with 46 for the boosted comparator;
both call many 0.2% and 1% points `minimal`. These results do not show a
dependable overall benefit from the CNN's patch inputs. The Santa Cruz figures
use the corrected USGS 3DHP water-distance input while retaining the same 450
point locations and labels; they supersede the initial scores.

The Santa Cruz sample was intentionally balanced and may contain spatially
correlated points in narrow 0.2% polygons. Its class frequencies do not
measure real-world prevalence. The Bay Area test blocks and Santa Cruz region
have appeared in prior model comparisons, so further tuning against these
results would need a fresh geographic holdout. Neither dataset has a
levee-related or otherwise `Unavailable` target. The model should not replace
the app's unavailable FEMA benchmark.

See the [Bay Area evaluation](../reports/terrain-three-class-cnn-evaluation.md)
and [Santa Cruz evaluation](../reports/terrain-three-class-cnn-external-evaluation.md)
for per-class metrics and confusion matrices.
