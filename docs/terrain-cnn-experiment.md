# Point-centered 2D CNN experiment

This is an offline comparison for FEMA mapped `one_percent` versus `minimal`
classes. It does **not** predict observed flooding during a storm. The app can
display its research score through the model selector.

## Inputs and model

Each calibration latitude/longitude is projected to California Albers
(`EPSG:3310`) and centers a 64 × 64 patch at 30 m resolution. The seven image
channels are slope, local elevation at 250 m and 500 m, log flow accumulation,
height above drainage, elevation relative to the point, and NLCD
imperviousness. Two additional channels mark valid topography and
imperviousness pixels. The DEM and NLCD are resampled onto the existing 30 m
terrain grid. FEMA polygons and map classes are never image channels.

Three two-convolution blocks encode the patch; pooled features are combined
with the same ten point inputs as the terrain XGBoost model. The comparison
fits XGBoost on precisely the same buffered training rows. The two CNN
candidates use ordinary loss or a 1.5× weight for `minimal` rows below 50 m.
All other training rows stay in both candidates.

## Reproduce

The patch builder needs Python with `osgeo` (GDAL) and NumPy. The training
environment needs the packages in `requirements-cnn.txt`. On this macOS
workspace, Homebrew `python3` provides GDAL and `.venv-model` holds the model
packages.

```sh
.venv-model/bin/python -m pip install -r requirements-cnn.txt
npm run model:cnn:patches
npm run model:experiment:terrain-cnn
```

The patch command writes aligned raster and patch caches to
`data/cache/terrain-cnn/`; source rasters are described in `data/README.md`.
For an existing checkpoint, `npm run model:cnn:rasters` prepares only the two
aligned rasters needed by the app and skips the large training patch array.
Live CNN scoring also needs the trusted `models/flood-risk-terrain-cnn.pt`
checkpoint and a `.venv-model` environment with `requirements-cnn.txt`.
The selector fetches its score on demand; if any of these local dependencies
are missing, the app displays an unavailable message for the CNN.
The training command fits two CNN candidates, selects an epoch and candidate by
validation `minimal` false-positive rate at at least 95% `one_percent` recall,
then evaluates the selected model once on test. It writes
`reports/terrain-cnn-evaluation.{json,md}` and the research-only
`models/flood-risk-terrain-cnn.pt` checkpoint.

## Evaluation limits

The existing geographic partitions are used, with a 1,920 m axis-aligned
buffer excluding training points whose image patches could overlap validation
or test patches. Validation patches near test patches are excluded. This
changes the training set, so the script retrains the tabular baseline on the
same rows and chooses both thresholds on validation using the same recall
rule. The test partition has appeared in earlier model comparisons and is not
a fresh independent holdout. The results measure FEMA map agreement, not
inundation at `minimal` locations during historical events.

## Result

The unweighted CNN at epoch 10 was selected. At thresholds chosen on
validation for at least 95% `one_percent` recall, it made 68/812 `minimal`
false positives on test, compared with 132/812 for buffered XGBoost. It also
detected 328/342 `one_percent` test points, compared with 338/342 for XGBoost,
and had a worse Brier score (0.045 versus 0.031).

The lower false-positive count is largely an operating-threshold tradeoff.
At the **same test recall** of 328/342, a post-hoc ranking diagnostic yields
55 CNN versus 54 XGBoost `minimal` false positives. Those test-derived
thresholds cannot be used for deployment. The experiment does not establish
that the spatial CNN discriminates FEMA classes better than the tabular model.
The complete metrics and elevation slices are in
[`reports/terrain-cnn-evaluation.md`](../reports/terrain-cnn-evaluation.md).

An [extended training run](../reports/terrain-cnn-extended-comparison.md) lowers
the learning rate to 0.0003, allows up to 80 epochs, and stops after 15 epochs
without validation improvement. Its selected checkpoint improves the
validation `minimal` false-positive count from 89 to 83, but on the Bay Area
test detects 324/342 `one_percent` points instead of 328/342 with nearly the
same `minimal` false-positive count (67 versus 68). The original checkpoint
remains in the app.

## Northern Santa Cruz external test

The saved CNN was also scored, without retraining or threshold changes, on
the separate northern Santa Cruz FEMA sample. Its 64 × 64 patches came from
the same aligned rasters as the Bay Area experiment. The CNN's balanced
accuracy fell from 0.938 on the Bay Area test to 0.837 on 300 Santa Cruz
minimal/1% points. At the original threshold it detected 101/150 Santa Cruz
1% points and produced 0/150 minimal false positives, compared with 328/342
and 68/812 respectively in the Bay Area test. Santa Cruz ROC AUC was 0.997,
so the saved score ranking remained strong even though the original decision
threshold transferred poorly. This is an inference from the external scores;
the threshold was not adjusted on Santa Cruz labels.

The first Santa Cruz pass used Bay-centered hydrography that omitted nearby
San Lorenzo River flowlines. It overstated nearest-water distance at the
sampled 1% locations and produced 53/150 detections. The current dataset
adds official USGS 3DHP flowlines while keeping all 450 sample coordinates
fixed; the 101/150 result above supersedes that first pass. The remaining
misses tend to sit higher above the derived drainage surface and on less
depressed terrain within the mapped 1% polygons.

Run `python3 scripts/build-santa-cruz-north-cnn-patches.py` and then
`.venv-model/bin/python scripts/evaluate-santa-cruz-north-terrain-models.py`
to reproduce the external check. The [external evaluation report](../reports/santa-cruz-north-fema-terrain-external-evaluation.md)
includes the CNN and tabular model confusion matrices.

A [point-only ablation](../reports/terrain-cnn-point-only-ablation.md) trains the
same tabular branch and output head without image pixels on the buffered rows.
After allowing it 100 epochs, the point-only model produced 115/812 `minimal`
false positives at 333/342 `one_percent` recall on test, versus 68/812 at
328/342 for the CNN. At the same post-hoc test recall of 328/342, the counts
were 73 versus 55. This supports testing the image branch further, but removing
it also changes model capacity. A patch-shuffled CNN using the same architecture
is a tighter control. A genuinely new geographic holdout is needed for
confirmation. Observed event labels are needed to test whether either model
predicts flooding in FEMA `minimal` areas.
