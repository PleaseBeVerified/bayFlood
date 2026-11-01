# DNN data

This directory contains the reproducible tabular inputs and outputs for the
BayFlood DNN. Raw geospatial datasets are excluded from Git because they are
large; follow the paths below when preparing a local copy.

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
`npm run dnn:train:three-class` if continuing that offline study. The blank
imperviousness template is optional and can be regenerated with
`npm run dnn:data:template`.

The full DNN feature definitions, target limitations, geographic split, and
2056 scenario assumptions are documented in
[`../docs/dnn-data-pipeline.md`](../docs/dnn-data-pipeline.md).
