# Training Data

Use this folder for machine-learning training inputs and generated tables.

## Files

- `training-labels.csv`: Hand-entered or imported labeled locations. Edit this
  file first.
- `training-table.csv`: Generated feature table for model training. Do not edit
  this by hand once a build script exists.
- `physical-hazard-calibration.csv`: Spatially sampled FEMA hazard classes and
  non-FEMA physical inputs for the experimental unknown-region model.

## Label Format

Each row in `training-labels.csv` should represent one location.

```csv
id,lat,lng,flooded,source,flood_year,notes
example-1,37.5301,-122.2730,1,manual,2017,"Known flood report"
example-2,37.7749,-122.4194,0,random_sample,,"Comparison point"
```

Set `flooded` to `1` for a known flood/claim/report location and `0` for a
comparison location without a known flood label.

Set `flood_year` for known events when it is available. Keep it empty for
comparison rows. The table builder can also infer a four-digit event year from
legacy NFIP notes. Flood year is retained for temporal auditing and future
survival analysis; it is deliberately excluded from the current location model
because it is unknown at prediction time.

The generated `training-table.csv` adds app-derived features such as elevation,
FEMA zone, SFHA distance, rainfall, and nearby-water distance.

## Build The Training Table

After adding labeled points to `training-labels.csv`, run:

```sh
npm run training:table
```

This writes `training-table.csv`.

## Train The Baseline Model

After rebuilding `training-table.csv`, run:

```sh
npm run training:model
```

This trains a small dependency-free logistic regression model and writes:

```text
models/flood-claim-logistic-model.json
```

The model uses independent physical inputs from the BayFlood feature pipeline:

- elevation
- FEMA zone and subtype
- Special Flood Hazard Area flag and distance
- NOAA 100-year, 24-hour rainfall
- nearby-water type and distance

Missing FEMA membership and distance values remain missing in the feature
table and are median-imputed by the trainer. They are not converted to a
confirmed non-SFHA value or an arbitrarily large distance. This allows the
model to screen locations with incomplete FEMA mapping without automatically
treating missing data as low risk.

Elevation, distance, rainfall, and SFHA inputs are direction-constrained so
they cannot learn physically contradictory effects.

The target is still `flooded`, so the model should be interpreted as estimating
whether a location resembles known NFIP claim points in this dataset, not as a
literal probability that a specific parcel will flood.

The trainer holds all rows with the same rounded coordinate in the same split.
This prevents duplicate redacted NFIP locations from leaking between training
and validation data.

## Physical Unknown-Region Calibration

Build a spatial sample from FEMA-mapped Bay Area regions, then train the
physical-only ordered gradient-boosted model. Create its Python environment
once before training:

```sh
python3 -m venv .venv-model
source .venv-model/bin/activate
pip install -r requirements-model.txt
```

Then run:

```sh
npm run calibration:physical-table
npm run calibration:physical-model
```

Or run both stages:

```sh
npm run calibration:physical
```

The target labels are FEMA mapped-event classes, but the prediction features
are limited to elevation, rainfall, nearest-water context, slope, local
elevation at 250 m and 500 m, D8 flow accumulation, and height above modeled
drainage. FEMA zone, floodplain distance, claims, policy history, address
history, and the separate claim-similarity score are excluded. The default
generator creates 10,000 rows. Deterministic 0.05-degree geographic blocks are
split into training, validation, and untouched test partitions so nearby rows
cannot be randomly leaked across evaluation sets.

Training and production both sample the 30 m multi-band runtime COG created by
`npm run terrain:runtime`. This prevents a train/serve mismatch. The original
10 m rasters remain the source for regenerating that deployment artifact.

The generated artifact is `models/flood-physical-hazard-model.json`. Its
dependency-free tree representation is evaluated directly by the Next.js API;
Python and XGBoost are only required during training. Its
central estimate uses annual anchors of 1% for SFHA terrain, 0.2% for moderate
terrain, and an explicitly assumed 0.1% midpoint for minimal-hazard terrain.
The API also returns a range that varies the minimal anchor from 0% to 0.2%.

To benchmark a class-balanced model that predicts the three FEMA-equivalent
classes directly and calibrates its probabilities on separate geographic
blocks, run:

```sh
npm run calibration:three-class-benchmark
```

This writes `reports/three-class-model-evaluation.md` and separate benchmark
model artifacts. It does not replace the production model. Hyperparameter
tuning, vector-scaling calibration, and final testing use mutually exclusive
geographic blocks.

Generate 1,500 additional, nonduplicate 0.2%-class rows without sampling the
untouched test blocks:

```sh
npm run calibration:physical-table -- --count=1500 --hazard=point_two_percent --partitions=training,validation --exclude=data/physical-hazard-calibration.csv --output=data/physical-hazard-point-two-extra.csv
```

This intentionally writes a separate reviewable CSV. Do not overwrite or merge
it into the main calibration table until its class counts, coordinates, county
coverage, and geographic-block distribution have been audited.

After auditing, create a deduplicated expanded table while preserving the
original table and untouched test blocks:

```sh
npm run calibration:merge-point-two
```

Then train a separately named expanded-data benchmark:

```sh
npm run calibration:three-class-expanded
```

The merge removes added locations within 30 meters of an existing or already
retained location. It writes `data/physical-hazard-calibration-expanded.csv`
and a row-level audit to `reports/physical-hazard-point-two-merge.json`.

Search added-data amounts (0%, 25%, 50%, 75%, 100%), class-weight strengths,
and tree settings without using the untouched test set for selection:

```sh
npm run calibration:three-class-search
```

Only added training-block rows participate in the search. The original tuning
and calibration blocks retain their natural class distribution, and the final
test set is evaluated once after all choices are frozen.

## 100-Year Output

The app reports values that must not be confused:

- **Primary 100-year result:** the FEMA benchmark when available; otherwise an
  experimental physical-only estimate with a range.
- **Model exposure signal:** the 0-100 similarity to known NFIP claim
  locations. It supports addresses with no known history, but it is not a
  time-based probability.
- **FEMA 100-year event benchmark:** `63.4%` for a mapped 1%-annual-chance
  event and `18.1%` for a mapped 0.2%-annual-chance event, calculated as
  `1 - (1 - annual chance)^100`.

The physical-only result is a FEMA-equivalent terrain estimate, not a verified
parcel probability. A literal property probability will still require verified
never-flooded labels with observation start/end years and a calibrated
time-to-event model.

## Prepare Property Observation Data

Use `property-observations.csv` for parcel- or building-resolved event and
non-event histories. Rows marked `pending` are candidates only and are excluded
from training. A `verified` row must provide a defensible observation interval,
traceable evidence, and precise coordinates.

```sh
npm run hazard:prepare
```

This validates the observations, samples the same physical features used by
the app, writes `property-observation-features.csv`, expands verified intervals
into `property-hazard-table.csv`, and generates the calibration-readiness
report in `reports/property-hazard-readiness.md`.

Public FEMA candidates can be generated automatically:

```sh
npm run training:openfema
```

This downloads OpenFEMA v3 claims and policy exposures across the nine Bay Area
counties. The output remains `pending` because public FEMA locations are
redacted and too coarse for automatic parcel matching. It reduces manual data
collection but does not fabricate verified address-level outcomes.

## Complete Area-Rate Data

Run the complete, resumable public-data aggregation and address join:

```sh
npm run training:area-pipeline
```

This produces `fema-area-year-rates.csv`, `fema-area-rates.csv`, and
`property-area-features.csv`. The rate denominator is calendar-year NFIP policy
exposure from the complete Bay Area policy extract; the numerator is claims in
the same years and Census block groups. Each historical area-year feature uses
only earlier years and is shrunk toward its prior county rate.

These values are suitable as an area-level historical insurance feature. They
are not verified parcel flood histories or 100-year property probabilities.
The final property model remains gated by `reports/property-hazard-readiness.md`.

## Test Against First Street

Keep independent First Street checks in `firststreet-test-cases.csv`. These
rows are never imported into the training table, which prevents test leakage.

```sh
npm run testing:firststreet
```

The benchmark compares low (`1-3`) versus elevated (`4-10`) risk classes and
prints the trained-model score. Add many properties across all Flood Factor
values before treating the result as an accuracy estimate. The current two
minimal-risk properties are regression checks, not a representative external
validation set.

## Generate Comparison Labels

After downloading and unzipping the Census county shapefile into
`data/tl_2025_us_county/`, run:

```sh
npm run training:comparisons -- --count=30
```

This appends generated `flooded = 0` rows to `training-labels.csv`. The script
keeps manually entered rows and replaces only rows whose IDs start with
`comparison-generated-`.

Comparison points can be generated as a mixed set, hill-only set, or
lowland-only set:

```sh
npm run training:comparisons -- --count=500 --profile=mixed
npm run training:comparisons -- --count=250 --profile=hill
npm run training:comparisons -- --count=250 --profile=lowland
```

Use `--dry-run` to preview rows without editing `training-labels.csv`:

```sh
npm run training:comparisons -- --count=20 --profile=mixed --dry-run
```

The `mixed` profile creates half hill comparison points and half lowland
comparison points. Hill comparisons require elevation of at least 60 meters.
Lowland comparisons require elevation from 1 to 15 meters, outside the mapped
SFHA, outside coastal V/VE zones, outside mapped open water, and within 3,000
meters of an SFHA.

Hill generated points must be at least 5,000 meters from known NFIP claim
labels. Lowland generated points must be at least 1,000 meters from known NFIP
claim labels so the model can learn from Bay-flat comparison locations without
excluding most of the shoreline geography.
