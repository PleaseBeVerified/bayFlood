# DNN data

This directory contains the reproducible tabular inputs and outputs for the
BayFlood DNN. Raw geospatial datasets are excluded from Git because they are
large; follow the paths below when preparing a local copy.

## Experimental terrain-coverage weights

`terrain-coverage-training-weights.csv` contains experimental weights for the
5,521 existing Bay Area training IDs retained after the three-class CNN's
1,920 m geographic buffers. Its adjacent `.source.json` records the source
checksums, recipe checksum, and training-only role. The binary weight column
is populated only for the 4,982 minimal/1% rows. No external-test rows are
included and no labels are changed.

Run `.venv-model/bin/python scripts/investigate-terrain-training-coverage.py`
to reproduce the weights and coverage audit. Add `--pilot` to run paired
boosted/DNN controls and coverage weighting across three seeds; that experiment
writes reports and does not replace saved models. The separate
`.venv-model/bin/python scripts/train-terrain-cnn-coverage-pilot.py` experiment
runs three paired seeds for binary and three-class CNNs on the same buffered
training IDs. Its results are in
`../reports/terrain-cnn-coverage-weighting-pilot.md`; experiment checkpoints
stay under the ignored `data/cache/terrain-cnn/coverage-weighting/` directory.
Full collection, sampling, and validation guidance
is in [the terrain generalization plan](../docs/terrain-generalization-training-plan.md).

## Northern Santa Cruz FEMA terrain external test set

Run `python3 scripts/fetch-santa-cruz-north-fema-zones.py` and
`python3 scripts/fetch-santa-cruz-north-3dhp-flowlines.py`, then
`python3 scripts/build-santa-cruz-north-fema-terrain-sample.py` to reproduce
`santa-cruz-north-fema-map-class-terrain-sample.csv`. The fetcher uses the
[official FEMA NFHL Flood Hazard Zones layer](https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28)
for Santa Cruz County map `06087C` within the northern study window
(-122.16, 37.005, -121.99, 37.155). Its raw GeoJSON is ignored by Git;
the sample's adjacent `.source.json` records the source checksum, study bounds,
feature sources, and sampling rules.

Water distance uses the existing local hydrography plus the
[USGS 3D Hydrography Program flowline layer](https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer/50)
within and around this study area. The earlier Bay-centered flowline files
omitted the nearby San Lorenzo River at some Felton points and overstated
distance to water by kilometers. The 450 selected coordinates were preserved
when the 3DHP source was added; only their water-distance input changed.
The source GeoJSON and checksum manifest are under `../hydrography/`.

The CSV has 450 deterministic, balanced points: 150 each labeled `minimal`,
`point_two_percent`, and `one_percent` in `mapped_hazard_class`. Every row
also carries its raw FEMA zone, zone subtype, source polygon ID, and map ID.
All rows have valid elevation, imperviousness, design-storm rainfall, five
terrain fields, and a local water feature within 5 km. Points are at least
10 m inside a FEMA polygon and 30 m apart. The `spatial_partition` column
is `external_test` for every row: keep the entire region out of existing-model
training and tuning when measuring geographic transfer. If this sample is
later used for training, create a different independent region for testing.
The original point selection still required a feature within 5 km in the
older local water layers; this legacy eligibility condition limits how
representative the sample is across the whole study window.

These labels describe FEMA **mapped annual-chance hazard**, not observed
inundation or verified dry locations in a particular storm. The balanced
sample is suitable for classwise model comparison; its class frequencies do
not estimate real-world flood-zone prevalence. Narrow 0.2% zones may still
produce spatially correlated points despite the separation rule.

For the CNN, run `npm run model:cnn:rasters` if the aligned raster cache is
missing, then `python3 scripts/build-santa-cruz-north-cnn-patches.py` to
extract 64 × 64 patches for the 300 explicit minimal and 1% rows. Run
`.venv-model/bin/python scripts/evaluate-santa-cruz-north-terrain-models.py`
to score the existing three-class, binary terrain, five-input, and CNN models
on this holdout without retraining. The binary comparisons exclude the 0.2%
rows because those models have no 0.2% target. The separate three-class CNN
and its buffered boosted comparator use the same corrected dataset through
`.venv-model/bin/python scripts/evaluate-terrain-three-class-cnn-external.py`.
Results and classwise confusion matrices are in
`../reports/santa-cruz-north-fema-terrain-external-evaluation.md` and its
adjacent JSON file, plus
`../reports/terrain-three-class-cnn-external-evaluation.md`.

## Tracy–Lathrop–Stockton FEMA terrain external test set

Run `python3 scripts/fetch-tracy-lathrop-stockton-sources.py`, then
`python3 scripts/build-tracy-lathrop-stockton-fema-terrain-sample.py` to
reproduce `tracy-lathrop-stockton-fema-map-class-terrain-sample.csv`.
The source fetch records checksums and retrieval dates beside 465 polygons
from the [FEMA NFHL Flood Hazard Zones layer](https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28)
for San Joaquin map `06077C`, and 9,200 flowlines from the
[USGS 3DHP layer](https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer/50).
The study window is `(-121.55, 37.68, -121.15, 37.98)` in WGS84.

The CSV contains 450 deterministic, model-input-complete points: 150 each in
FEMA mapped `minimal`, `point_two_percent`, and `one_percent` classes.
The `tls-fema-terrain-*` IDs, `San Joaquin County` county value, and
`external_test` partition distinguish it from Bay Area training and Santa
Cruz evaluation data. Source polygon IDs, raw zones, and the map ID remain
on every row. Points are at least 10 m inside a FEMA polygon and 30 m apart.
Unlike the first Santa Cruz sample, the 3DHP water network was included
before point selection as well as water-distance calculation.

For CNN scoring, run the patch builder with `--region tracy-lathrop-stockton`
and repeat with `--three-class` to prepare
the separate ignored patch caches. Then run
`.venv-model/bin/python scripts/evaluate-tracy-lathrop-stockton-terrain-models.py`
and `.venv-model/bin/python scripts/evaluate-tracy-lathrop-stockton-three-class-cnn.py`.
The reports are
`../reports/tracy-lathrop-stockton-fema-terrain-external-evaluation.md` and
`../reports/tracy-lathrop-stockton-three-class-cnn-external-evaluation.md`.
No saved model is retrained or threshold-tuned on this region. The 0.2%
points are excluded from binary scoring because those models have no 0.2%
target. These labels are FEMA mapped annual-chance classes, not observed
flood-event outcomes or verified dry locations. Balanced, spatially sampled
classes do not measure regional prevalence, and source polygons can induce
correlated errors.

## Sacramento County pilot datasets

Run `python3 scripts/fetch-sacramento-flood-zones.py` to download [Sacramento
County's public FEMA 100/500-year floodplain layer](https://mapservices.gis.saccounty.net/arcgis/rest/services/DWR_HYDROLOGY/MapServer/7) to
`FEMAdata/sacramento_county_100_500_year_floodplain.geojson`. The script also
writes a source manifest beside the GeoJSON with the retrieval time, feature
count, field list, coordinate system, and SHA-256 checksum. Both files are local
raw geospatial data and are excluded from Git. The query rounds coordinates to
six decimals and requests a 0.00001-degree maximum geometry offset to keep the
download tractable; use the official FEMA map for boundary decisions.

The layer is a mapped hazard product. It does not provide event observations,
verified dry locations, or the elevation, rainfall, water proximity, terrain,
and imperviousness features needed by the calibration table. It is therefore
kept separate from the Bay Area training CSV and model until those inputs and
an out-of-region validation plan are prepared.

Build the separate Sacramento map-class sample with
`python3 scripts/build-sacramento-fema-sample.py`. The tracked
`sacramento-fema-map-class-sample.csv` contains 3,000 deterministic points:
1,000 each from mapped minimal, 0.2%, and 1% classes. Each point is at least
30 m inside a source polygon and includes the source polygon ID, 2025 NLCD
imperviousness, and NOAA Atlas 14 design-storm rainfall. Its `.source.json`
file records the source checksum and sampling method. This is a FEMA **map
class** sample, not an observed flood sample or a complete model training table.
The local elevation and terrain rasters have no valid values at these 3,000
points, and water-distance features have not been joined.

For historical evidence from the December 2022–January 2023 storms:

- `python3 scripts/build-sacramento-storm-reports.py` extracts 34 Sacramento
  County Flood and Flash Flood reports from [NOAA's Storm Events bulk
  archives](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/)
  into `sacramento-reported-floods-2022-2023.csv`. The reports include dates,
  locations, causes, and narratives. Their coordinates describe incidents or
  affected areas; they are not surveyed flood footprints.
- `python3 scripts/fetch-sacramento-river-observations.py` downloads 15,084
  [USGS instantaneous stage and discharge
  readings](https://waterservices.usgs.gov/docs/instantaneous-values/) from the
  Cosumnes River at Michigan Bar, American River at Fair Oaks, and Sacramento
  River at Freeport into `sacramento-river-observations-2022-2023.csv`.
  These are station observations and event inputs, not property-level outcomes.

Both historical CSVs have adjacent `.source.json` manifests. NOAA bulk archives
and the raw USGS response are cached under the ignored `data/cache/` directory.
No verified dry observations or surveyed inundation coverage were found in
these sources, so these files must not be used as positive/negative training
labels for an event-inundation model.

## Required source raster

Place the full-resolution 2025 Annual NLCD fractional-imperviousness GeoTIFF
here:

```text
data/nlcd/fractional-impervious-2025-full.tiff
```

Then sample the DNN locations:

```sh
npm run dnn:data:sample-imperviousness
```

## Current model files

- `physical-hazard-calibration.csv` — source locations and FEMA-derived labels.
- `dnn-imperviousness.csv` — current imperviousness sampled from Annual NLCD.
- `dnn-training-table.csv` — the only default training CSV, used by both the
  current boosted model and the DNN comparison.

Build the tables and train the model with:

```sh
npm run dnn:data
npm run dnn:train
npm run model:train:boosted
```

The optional `npm run dnn:data:research` command also writes the three-class
training CSV and unlabeled 2056 scenario CSV. Run it before
`npm run dnn:train:three-class`, `npm run dnn:train:terrain-three-class`,
or `npm run model:experiment:terrain-three-class`
if continuing that offline study. The blank
imperviousness template is optional and can be regenerated with
`npm run dnn:data:template`.

The full DNN feature definitions, target limitations, geographic split, and
2056 scenario assumptions are documented in
[`../docs/dnn-data-pipeline.md`](../docs/dnn-data-pipeline.md).
