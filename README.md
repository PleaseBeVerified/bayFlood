# Bay Area Flood Risk Analyzer

The Bay Area Flood Risk Analyzer is a Bay Area flood-risk **screening** app.
Search for an address or select a point on the map to see current annual,
10-year, and 20-year FEMA mapped flood-event benchmarks, followed by a
separate terrain model score. You can switch between a boosted two-class model,
a two-class terrain DNN, and a research two-class terrain CNN.

> This is an educational research prototype. A mapped flood-event benchmark is
> not a parcel-level determination, engineering study, insurance quote, or a
> prediction that a building will flood.

## Displayed percentages

The **boosted** and **terrain DNN** scores measure similarity to the FEMA 1%
training pattern. The **terrain CNN** uses a 64 × 64 local raster patch and the
ten point inputs to score similarity to the same map pattern. Model scores
are displayed out of 100, **not** as percent chances
that a property floods. If elevation or terrain data is unavailable, the model
outputs are unavailable while any FEMA benchmark remains visible.
When the boosted model's higher/lower pattern classification disagrees with a FEMA
1%-annual-chance or minimal-hazard classification, the app calls out the
disagreement beside the score. The FEMA map remains the basis for the
displayed percentages.

For a point in a FEMA mapped 1%-annual-chance flood hazard area, the app shows
1.0% annually, 9.6% over 10 years, and 18.2% over 20 years. For a mapped
0.2%-annual-chance area, it shows 0.2%, 2.0%, and 3.9%. Shaded Zone X areas
specifically marked as 1%-annual-chance shallow flooding are also
shown with the 1% benchmark; they are identified separately from Special Flood
Hazard Areas. The multi-year calculation is
`100 × [1 − (1 − annual chance)^years]`, assuming the annual chance remains
constant and years are independent. For unshaded Zone X (minimal mapped
hazard), outside the 0.2%-annual-chance floodplain, the app shows map-based
upper bounds of `<0.2%` annually, `<2.0%` in 10 years, and `<4.0%` in 20 years,
rounded outward. Other flooding can occur
in these areas. The values do not describe the chance of property damage or
interior flooding. The app labels a percentage "Unavailable" where the map
supplies no usable threshold (such as Zone D or a levee-related Zone X
designation) or neither source covers the point. This does
not mean zero flood risk.

The flood-zone lookup first queries FEMA's live NFHL flood-hazard-area layer.
When the live service is unavailable or has no matching polygon, it falls back
to the local GeoJSON files in `FEMAdata/` and labels that source separately.
If neither source matches, the benchmark remains unavailable rather than being
treated as minimal risk. Check
the current [FEMA Flood Map Service Center](https://msc.fema.gov/portal/home)
before using a result for decisions. This benchmark does not project climate
change, sea-level rise, drainage changes, or changes to flood defenses.

## Research model

All three app models use ten physical point inputs:

- elevation in meters;
- 2025 NLCD fractional impervious surface (%);
- NOAA 100-year, 24-hour rainfall (mm);
- whether mapped water is available nearby; and
- log distance to the nearest mapped water feature (m);
- terrain slope (degrees);
- local elevation relative to the surrounding 250 m and 500 m areas;
- log flow accumulation (m²); and
- height above drainage (m).

The boosted model learns a binary FEMA-equivalent screening label: mapped
1%-annual-chance terrain versus minimal-hazard terrain. Moderate
(0.2%-annual-chance) rows are excluded from this binary benchmark. The terrain
DNN and CNN also compare the minimal and 1% classes, excluding 0.2% rows.
No model output is used for the displayed
10- or 20-year percentages.

The [terrain model evaluation](reports/terrain-fema-baselines.md) reports 91.5%
accuracy, 93.5% balanced accuracy, 0.873 F1, and 0.994 ROC AUC on the
geographically separated test split. The previous five-input boosted model
reached 94.4% balanced accuracy and 0.987 ROC AUC on that split. The binary
terrain DNN is selectable in the app; the previous five-input boosted model
remains available for research comparison.
These metrics measure agreement with FEMA-derived categories, not observed
property flooding.

A [ten-input terrain DNN](reports/terrain-dnn-evaluation.md) uses the same
features and geographic splits. After validation-based tuning, it reached
0.989 ROC AUC and 0.928 balanced accuracy on the geographic test set. Its
exported JSON is used by the app's two-class DNN option.

A separately trained three-class candidate now includes the 0.2%-annual-chance
mapped category. Its [held-out evaluation](reports/three-class-dnn-evaluation.md)
documents class-specific errors and a Newark spot check. It is an offline
research model and is not displayed for locations without FEMA data yet.
The [ten-input terrain three-class DNN](reports/terrain-three-class-dnn-evaluation.md)
and [terrain boosted classifier](reports/terrain-three-class-boosted-evaluation.md)
also use all 933 mapped 0.2% samples. The boosted candidate has the strongest
overall three-class macro F1 (0.768), but its 0.2% class F1 is only 0.528.
The three-class terrain DNN remains an offline research candidate. The app lets
you switch between the binary terrain boosted score, the binary terrain DNN
score, and the research terrain CNN. All describe
agreement with FEMA-derived map categories, not observed flooding at a property.

The [CNN experiment](docs/terrain-cnn-experiment.md) did not establish better
discrimination than the buffered boosted comparator at matched test recall.
Its score is available in the selector for research comparison.
A [point-only ablation](reports/terrain-cnn-point-only-ablation.md) found fewer
false positives for the CNN than for its tabular branch alone on the same
geographic test rows; a same-architecture patch-shuffle remains to be tested.
A separate [three-class CNN experiment](docs/terrain-three-class-cnn-experiment.md)
trains on the 0.2% samples. Its Bay Area geographic-test macro F1 was 0.742,
versus 0.749 for a boosted comparator trained on the same buffered points.
Both models transferred poorly to the balanced Santa Cruz sample; the
three-class CNN remains offline.

A separate [Tracy–Lathrop–Stockton external test](reports/tracy-lathrop-stockton-fema-terrain-external-evaluation.md)
uses 450 FEMA map-class points in western San Joaquin County with complete
terrain inputs and local USGS 3DHP flowlines. The [three-class CNN comparison](reports/tracy-lathrop-stockton-three-class-cnn-external-evaluation.md)
uses the same points. This region stays outside model training and tuning.

The project retains the unsourced 2056 input scenarios for research, but the
live app no longer presents their model scores as forecasts. The models are not
constrained to raise their scores as rainfall or paving increases.

## Run locally

```sh
npm install
python3 -m venv .venv-model
.venv-model/bin/pip install -r requirements-model.txt
npm run dev
```

Open `http://localhost:3000`.

The FEMA benchmark can use the local flood-zone GeoJSON files in `FEMAdata/`
when the live service has no match or is unavailable. These files are intentionally
not committed to GitHub. The current score and model
research pipeline additionally need:

- `elevation/` — USGS elevation GeoTIFF tiles
- `rainfall/rainfall_data.tif` — NOAA Atlas 14 rainfall raster
- `hydrography/` — mapped-water GeoJSON files
- `data/nlcd/fractional-impervious-2025-full.tiff` — full 2025 Annual NLCD
  fractional-imperviousness GeoTIFF
- `terrain/runtime/terrain-features-30m.tif` — five-band terrain raster used
  by the training table and live score
- `terrain/dem-10m.tif` — source elevation raster used to build the CNN grid

The terrain raster is ignored by Git. Any deployed app instance must have this
file at the same path or the terrain model score will be unavailable.

The CNN also needs `models/flood-risk-terrain-cnn.pt`, the aligned 30 m DEM and
imperviousness rasters under `data/cache/terrain-cnn/`, and a `.venv-model`
environment with `requirements-cnn.txt` installed. With the source rasters in
place, run `npm run model:cnn:rasters` to prepare the aligned grids. The CNN
option shows an unavailable message when these local resources are missing.

See [`data/README.md`](data/README.md) for the generated-table workflow.

## Rebuild the models

After placing the full NLCD raster in the location above:

```sh
npm run dnn:data:sample-imperviousness
npm run dnn:data
npm run dnn:train
npm run model:train:boosted
npm run model:experiment:terrain
npm run model:experiment:terrain-dnn
npm run test
```

Or run the current app pipeline with `npm run dnn:all`. To recreate the offline
three-class and scenario CSVs, run `npm run dnn:data:research` before
`npm run dnn:train:three-class`, `npm run dnn:train:terrain-three-class`,
or `npm run model:experiment:terrain-three-class`.

The app loads `models/flood-risk-terrain-full.json` for its boosted score
and `models/flood-risk-terrain-dnn.json` for its two-class DNN score.
`models/flood-hazard-terrain-three-class-dnn.json`, `models/flood-risk-boosted.json`,
`models/flood-risk-dnn.json`, and the Python joblib files remain available
for offline comparison. Training preserves geographic partitions to reduce
leakage between nearby locations.
The separate three-class artifact is `models/flood-hazard-three-class-dnn.json`.

## Project layout

- `src/app/` — Next.js app and API routes
- `src/lib/dnn-model.ts` — dependency-free DNN inference
- `src/lib/terrain-dnn-model.ts` — ten-input, two-class terrain DNN inference used by the app
- `src/lib/terrain-three-class-model.ts` — ten-input, three-class terrain DNN inference for offline comparison
- `src/lib/boosted-model.ts` — current terrain model inference
- `src/lib/terrain.ts` — runtime terrain-raster sampler
- `src/lib/fema-event-benchmark.ts` — mapped-zone lookup and 10/20-year calculation
- `src/lib/nlcd.ts` — imperviousness-raster sampler
- `scripts/` — model data preparation and training scripts
- `data/` — generated training tables and documented source-data placement
- `models/` — exported model artifacts
- `docs/dnn-data-pipeline.md` — data schema and scenario assumptions

## Data sources

The project combines Annual NLCD land-cover data, USGS elevation data, NOAA
Atlas 14 precipitation-frequency data, mapped hydrography, and FEMA-derived
hazard labels. Source licenses, dates, coverage, and preprocessing choices
should be recorded alongside any science-fair or journal submission.
