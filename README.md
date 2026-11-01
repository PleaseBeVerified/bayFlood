# Bay Area Flood Risk Analyzer

The Bay Area Flood Risk Analyzer is a Bay Area flood-risk **screening** app.
Search for an address or select a point on the map to see current annual,
10-year, and 20-year FEMA mapped flood-event benchmarks, followed by a
separate current-condition model score.

> This is an educational research prototype. A mapped flood-event benchmark is
> not a parcel-level determination, engineering study, insurance quote, or a
> prediction that a building will flood.

## Displayed percentages

The **current score** is the boosted model's measure of similarity to mapped higher-risk
areas in its training data. It is displayed out of 100, **not** as a percent
chance that a property floods. If elevation data is unavailable, the app
leaves the current score unavailable while still showing any FEMA benchmark.
When the model's higher/lower pattern classification disagrees with a FEMA
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

The current app model uses five physical inputs:

- elevation in meters;
- 2025 NLCD fractional impervious surface (%);
- NOAA 100-year, 24-hour rainfall (mm);
- whether mapped water is available nearby; and
- log distance to the nearest mapped water feature (m).

It learns a binary FEMA-equivalent screening label: mapped 1%-annual-chance
terrain versus minimal-hazard terrain. Moderate (0.2%-annual-chance) rows are
excluded from this benchmark. Its output is not used for the displayed
10- or 20-year percentages.

The [boosted model evaluation](reports/boosted-dnn-comparison.md) reports 93.8%
accuracy, 94.4% balanced accuracy, and 0.902 F1 on the geographically separated
test split. The earlier DNN achieved 90.1% accuracy, 92.1% balanced accuracy,
and 0.853 F1 on the same split. The DNN is retained for research comparison.
These metrics measure agreement with FEMA-derived categories, not observed
property flooding.

A separately trained three-class candidate now includes the 0.2%-annual-chance
mapped category. Its [held-out evaluation](reports/three-class-dnn-evaluation.md)
documents class-specific errors and a Newark spot check. It is an offline
research model and is not displayed for locations without FEMA data yet.

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

See [`data/README.md`](data/README.md) for the generated-table workflow.

## Rebuild the models

After placing the full NLCD raster in the location above:

```sh
npm run dnn:data:sample-imperviousness
npm run dnn:data
npm run dnn:train
npm run model:train:boosted
npm run test
```

Or run the current app pipeline with `npm run dnn:all`. To recreate the offline
three-class and scenario CSVs, run `npm run dnn:data:research` before
`npm run dnn:train:three-class`.

The app loads `models/flood-risk-boosted.json` for its current-condition pattern
score. `models/flood-risk-dnn.json` and the Python joblib files remain available
for offline comparison. Training preserves geographic partitions to reduce
leakage between nearby locations.
The separate three-class artifact is `models/flood-hazard-three-class-dnn.json`.

## Project layout

- `src/app/` — Next.js app and API routes
- `src/lib/dnn-model.ts` — dependency-free DNN inference
- `src/lib/boosted-model.ts` — current app model inference
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
