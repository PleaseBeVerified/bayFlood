# DNN tabular data pipeline

This is the reproducible data-preparation stage for BayFlood's tabular deep
neural network (DNN). The illustrative 2056 scenarios are sensitivity cases,
not forecasts.

## Why there are two tables

The current-condition table contains both inputs and a current FEMA-equivalent
binary label. The 2056 table contains projected scenario inputs and deliberately
has no label. Pairing projected inputs with current labels would mix time
periods and train the network on a target that does not describe those inputs.

The original binary training target is:

- `1`: FEMA 1%-annual-chance mapped class.
- `0`: FEMA minimal-hazard mapped class.
- FEMA 0.2%-annual-chance rows are excluded from the first binary benchmark.

The label remains a FEMA-equivalent physical-hazard class, not an observed
property flood outcome.

The optional `dnn-three-class-training-table.csv` includes all three mapped classes,
including the 0.2%-annual-chance category. It keeps the same location-level
geographic partitions as the binary table. The three-class candidate is trained
separately and is not yet used to fill gaps in the live app.

## Supply current imperviousness

Place the full 2025 Annual NLCD fractional-imperviousness GeoTIFF at
`data/nlcd/fractional-impervious-2025-full.tiff`, then sample the locations:

```sh
npm run dnn:data:sample-imperviousness
```

This writes `data/dnn-imperviousness.csv` for all calibration locations. The
sampler records its data source, and the builder rejects blank, out-of-range,
duplicate, unknown, or unsourced values.

`location_id` is a join key only. It is not a DNN input.

## Build the current training table

Run:

```sh
npm run dnn:data
```

The default command writes:

- `data/dnn-training-table.csv`: current inputs, binary label, and deterministic
  geographic partition.
- `reports/dnn-data-readiness.json`: counts by target and geographic split.

Run `npm run dnn:data:research` to also write the optional three-class training
table and unlabeled 2056 scenario table. The default build keeps one training
CSV in the workspace.

Rainfall is converted from the existing NOAA 100-year, 24-hour estimate in
inches to millimeters with `inches * 25.4`. The existing nearest-water value can
represent several mapped water types, so the output calls it nearest water
rather than implying that every feature is a river or drain. Locations without
a mapped nearby-water result retain a blank distance and receive
`nearest_water_available=0`; a future trainer must fit any distance imputation
on training rows only and keep this availability indicator as an input.

## Projection assumptions

`data/dnn-scenarios.json` versions the low, medium, and high sensitivity cases.
For each case:

```text
projected imperviousness = min(100, current imperviousness + percentage points)
projected rainfall = current rainfall * (1 + increase percent / 100)
```

Elevation and nearest-water distance remain unchanged in this simplified
stage. Every scenario row is marked
`illustrative_sensitivity_scenario_not_validated_forecast`. Replace the
configuration with sourced assumptions before presenting a scenario as a
planning projection.

The geographic split assigns complete 0.05-degree blocks to training,
validation, or untouched test partitions. Train the exported DNN with:

```sh
npm run dnn:train
```

The trainer uses these supplied partitions rather than randomly splitting rows,
and exports `models/flood-risk-dnn.json` plus a joblib model for offline
comparison. The current app uses a boosted classifier trained from the same
table and geographic partitions:

```sh
npm run model:train:boosted
```

Its export is `models/flood-risk-boosted.json`, with evaluation in
`reports/boosted-dnn-comparison.md`. The DNN remains a research benchmark.

Train the three-class candidate after creating its optional CSV:

```sh
npm run dnn:data:research
npm run dnn:train:three-class
```

It exports `models/flood-hazard-three-class-dnn.json` and joblib, and writes
`reports/three-class-dnn-evaluation.md`. Candidate selection uses validation
macro F1; the held-out test set is evaluated once after selection. The current
five features do not capture local drainage, levees, or building floor elevation,
so the class scores are not flood probabilities.
