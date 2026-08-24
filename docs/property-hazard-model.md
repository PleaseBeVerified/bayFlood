# Property-level flood hazard data

The next calibrated model must learn from property observation periods, not from a list of flood claims alone. A claim supplies an event, but a defensible annual hazard model also needs verified properties observed without an event so it knows the exposure time and can distinguish non-events from missing history.

## Workflow

1. Record parcel- or building-resolved observations in `data/property-observations.csv`.
2. Keep unverified candidates as `pending`; the validator excludes them from training.
3. Run `npm run hazard:prepare`.
4. Review `reports/property-hazard-readiness.md` before fitting or publishing a probability model.

Verified event rows require an event date and source. Verified censored (non-event) rows require evidence supporting the observation interval. Every verified row also requires a parcel or building identifier, a source URL, and coordinates precise to 30 meters or better.

The feature builder samples elevation, FEMA flood-zone context, 100-year 24-hour rainfall, and nearby water for each verified property. The hazard-table builder then creates one row per exposed calendar year, stopping at the first recorded event. This is the appropriate input shape for a discrete-time survival model.

## Probability interpretation

An annual hazard estimate `p` can be converted to a 100-year cumulative probability only under an explicit constant-hazard assumption:

`P(flood within 100 years) = 1 - (1 - p)^100`

That conversion must not be described as an accurate property probability until out-of-time and spatial holdouts demonstrate calibration. FEMA zone membership, claims history, and the current exposure score are inputs or benchmarks—not ground-truth 100-year probabilities.

BayFlood may display a separate experimental physical-only 100-year estimate
for addresses without a numeric FEMA classification. That model reconstructs
FEMA-equivalent hazard classes from elevation, rainfall, and water proximity;
it does not satisfy this property-observation readiness gate or replace a
property survival model.

The readiness thresholds are minimum gates for beginning calibration, not proof that the resulting model is accurate, suitable for insurance pricing, or a substitute for an engineering flood study.

## Automated public-data candidates

Run `npm run training:openfema` to download a bounded sample of current OpenFEMA v3 claim and policy records for all nine Bay Area counties. The default 250 records per county per dataset is the recommended reliable batch size. Larger `--max-per-county` values require pagination and may be throttled or time out on FEMA's policy endpoint; progress and retries are printed while the command runs.

The generator writes `data/openfema-observation-candidates.csv` and an audit report. Every generated record is deliberately marked `pending`: OpenFEMA removes street addresses and publishes coarse coordinates, so these rows cannot be assigned automatically to parcels or used as verified property outcomes. Policy terms indicate insured exposure, not proof that a property did not flood.

## Complete FEMA area-level signal

Run the complete public-data pipeline with:

```sh
npm run training:area-pipeline
```

The command downloads and caches the complete OpenFEMA v3 claims and policy Parquet files, filters all nine Bay Area counties, expands policy terms into calendar-year exposure, aggregates claims and policy-years by Census block group, and joins the result to the candidate-address file through the Census coordinate geography service. Downloads resume when interrupted and are skipped when the cached files are already complete.

The area-year signal uses only earlier years and applies credibility shrinkage toward the prior county rate. Rows carry time splits (`train` through 2019, `validation` for 2020-2022, and `test` from 2023) plus county-based spatial folds. The outputs are:

- `data/fema-area-year-rates.csv`: leakage-controlled model/evaluation rows.
- `data/fema-area-rates.csv`: one current historical signal per Census block group.
- `data/property-area-features.csv`: address candidates joined to the area signal.
- `reports/fema-area-signal-validation.csv`: time- and county-split validation totals.
- `reports/fema-area-rate-report.md`: coverage, validation, and limitations.

This solves scalable address enrichment: no one needs to look up each parcel manually to attach the public area signal. It does **not** create parcel labels. FEMA's public coordinates are redacted, NFIP policyholders are not a random sample of all properties, and a claim is not the same as every physical flood event. Therefore the pipeline deliberately reports `AREA_SIGNAL_READY__PARCEL_PROBABILITY_BLOCKED` and does not generate a 100-year property probability.

To join a different address CSV containing `lat` and `lng`, run:

```sh
npm run training:area-join -- --input=data/my-addresses.csv --output=data/my-address-area-features.csv
```
