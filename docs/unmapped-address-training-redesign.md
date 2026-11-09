# Training redesign for addresses without a usable FEMA benchmark

## Decision and current gap

The existing boosted model learns to reproduce a FEMA map distinction:
`one_percent` versus `minimal`. It has no observed flood outcomes. Replacing
`minimal` with a different name, rebalancing those rows, or using the three-class
DNN would still train against FEMA map categories. The app already computes its
separate pattern score when a FEMA lookup has no match, provided elevation can
be sampled; that score is not an observed flooding probability.

The chosen target is **event-conditioned inundation**: for a specified storm
or coastal-water-level event, estimate whether an outdoor location is
inundated at any time in that event window. In notation, the intended score is
`P(outdoor location inundated during event | location features, specified event
rainfall and water levels)`. This target can be checked against independent
observations and can give an unmapped address a meaningful score. A score for a
building interior or an annual probability requires additional outcome and
exposure data and must be validated separately.

The first implementation is a **scenario or hindcast model**: its rainfall and
water-level inputs describe the specified event. It is not automatically a
lead-time forecast. A forecast product would need archived forecasts available
at the chosen issue time for training and evaluation, with a fixed prediction
horizon and no later observations in its predictors.

Keep these states distinct at both training and inference time:

| FEMA lookup state | Meaning | Treatment |
| --- | --- | --- |
| Mapped 1% or 0.2% area | FEMA provides a mapped flood benchmark | Show it as map context; never use it as an event outcome. |
| Mapped minimal (unshaded Zone X) | Lower mapped hazard, not proof of no flooding | May be evaluated as a subgroup; never automatically label it dry. |
| Zone D or a special Zone X designation | Map does not give a usable numeric benchmark | Keep its specific designation; do not relabel it minimal. |
| No matching FEMA polygon | Map coverage/lookup gap | Eligible for model score if input coverage and validation support it. |
| FEMA service unavailable | Operational lookup failure | Retry or use a versioned local snapshot; do not interpret it as hazard evidence. |

## First training dataset: location × event

Each row represents a stable location during one historical event. The minimum
fields are:

| Group | Required information |
| --- | --- |
| Identity | `location_id`, coordinates, `event_id`, event start/end, study-area ID. |
| Label | `outcome` (`flooded`, `verified_dry`, or `unknown`), evidence type, observation time, source URI, footprint/survey version, positional uncertainty. |
| Static inputs | Elevation, slope, height above nearest drainage, flow accumulation, distance and type of water, imperviousness, coastal elevation or distance where applicable; each with source version and coverage flag. |
| Event inputs | Rainfall accumulated over defined event windows, antecedent rainfall, nearby river stage/flow, coastal water level/tide, and missingness flags. Record observation and forecast issue times so scenario/hindcast inputs cannot be confused with lead-time forecast inputs. |
| Context for evaluation | FEMA lookup status and mapped class, county, watershed, flood mechanism, and spatial block. Keep FEMA status out of the first model's predictors so it can be evaluated in unmapped areas. |

### Candidate inputs and availability

| Priority | Inputs | Current state | Next action |
| --- | --- | --- | --- |
| First | Event rainfall totals over fixed windows (for example, 6, 24, and 72 hours) and antecedent rainfall | The app has a NOAA **design-storm** raster, not event rainfall. | Join historical event rainfall from a dated observation product; use the same definitions for specified-storm inputs. |
| First | River stage/flow and coastal water level/tide relative to relevant local thresholds | No location-by-event table exists. | Match each event and location to appropriate observations; record station distance, datum, timestamps, and missingness. |
| First | Slope, height above drainage, flow accumulation, and local elevation at 250 m and 500 m | All 10,000 calibration rows contain these fields; `terrain/runtime/terrain-features-30m.tif` has five corresponding bands. The app does not currently sample this raster. | Verify raster definitions and coverage, then implement a sampler shared by training and inference. Transform skewed flow accumulation using training data only. |
| Next | Water-feature type and distance to coast, stream, or drainage separately | Nearest-water kind and distance exist in calibration data; the current model collapses them to distance plus availability. | Check source consistency and encode type and missingness without treating missing water as far away. |
| Later | Soil saturation, drainage infrastructure, levees, and building floor elevation | No consistent address-level source is integrated. | Add only where coverage, dates, and inference-time availability can be audited. |

Add features incrementally to the event-labeled baseline, comparing each group
on the same held-out storms and geographic areas. A feature must be calculable
for an unmapped address at scoring time. More inputs trained against the old
FEMA class labels can improve map imitation without improving observed-storm
predictions.

Use an observed inundation **footprint with documented survey coverage** for
the clearest labels: locations inside the footprint are positive; locations
outside it but inside the surveyed coverage are candidate verified negatives.
Review uncertainty at footprint edges and omit ambiguous points. A flood
complaint, high-water mark, or claim is positive evidence only at its supported
location and time. The absence of a report is `unknown`, not `verified_dry`.
Station-level water observations are event predictors; a high-water mark alone
does not establish a complete dry area around it.

### Ground-truth choices for FEMA-minimal locations

The FEMA `minimal` category is a sampling stratum, never an outcome label for
the event model. For each event, assign a location one of three outcomes:

1. **Flooded:** a field-verified inundation point, a well-located high-water
   mark, or a documented event footprint covering the location. Keep the
   evidence type because a point report and a mapped footprint have different
   spatial uncertainty.
2. **Verified dry:** a location inside the area actually inspected by a field
   survey or suitable event imagery, outside the observed inundation extent,
   with enough distance from uncertain boundaries and with evidence that water
   could have been detected at the observation time. The survey/imagery must
   cover the relevant event peak or another explicitly defined target time.
3. **Unknown:** no observation, no report, outside survey coverage, clouded or
   obstructed imagery, or an observation after water may have receded. Exclude
   this row from supervised loss and test metrics.

Official flood complaints, emergency calls, and insurance claims may add
positive examples if their locations and dates are verified. Their absence is
not a dry label. Public NFIP claims are too spatially coarse for individual
address labels. Satellite flood maps can supply broad wet extents and some dry
coverage, but sensor masks, buildings, vegetation, and overpass timing must be
checked before assigning negatives. Hydraulic or FEMA design-flood maps are
model outputs and should be used as context or weak labels, not as the
independent observed ground truth used to evaluate the new model.

The current repository contains no such labeled event table. Its 10,000-row
calibration CSV has FEMA-derived map labels. The property-observation template
has zero verified properties, and the OpenFEMA candidate workbook has zero
parcel-verified labels. Public NFIP claims can support area-level claim-rate
analysis with policy exposure denominators; their redacted locations and
insured-only population do not make address-level flood/dry labels.

Potential source inventory to check for the pilot area:

- [USGS Flood Event Viewer](https://www.usgs.gov/tools/flood-event-viewer): event high-water marks and sensor observations; use marks as positive evidence or as constraints on an independently produced footprint.
- [NOAA precipitation access](https://api.water.noaa.gov/about/precipitation-data-access): observed event rainfall, replacing the current static NOAA design-storm raster for event prediction.
- [NOAA coastal inundation dashboard](https://www.co-ops.nos.noaa.gov/inundationdb/): historical water levels and threshold exceedances at coastal stations; these are not address-level inundation labels.
- Local agency or USGS event-specific inundation polygons with explicit coverage metadata are preferred for the first labels. Record the exact dataset and version before joining rows.

A concrete pilot candidate is the [February 21, 2017 Coyote Creek flood
extent](https://assets.valleywater.org/files/2018-02/120617_IMC_AG.pdf)
published by Valley Water. Its map says the estimated flooded area came from
an aerial image of that event. First seek the underlying geospatial extent and
its aerial-image coverage; a PDF illustration alone is not precise enough to
assign address-level dry labels. The [Madrone gauge history](https://www.cnrfc.noaa.gov/obsRiver_hc.php?id=CYTC1&scale=0)
provides related event water-level context. One 2017 event could validate the
label-building workflow, but it cannot supply an unseen-event test by itself.

## Training and evaluation procedure

1. Inventory several events in one bounded subregion. Select events only when
   inundation evidence and observation coverage can be documented. Keep a
   source manifest with dates, units, spatial resolution, and license.
2. Build a reproducible location-by-event table. Deduplicate locations and
   events. Reject labels without evidence. For a lead-time forecast experiment,
   reject predictor values recorded after its issue time. Store `unknown` rows,
   but exclude them from supervised loss and test metrics.
3. Fit three baselines on the same labeled rows: a simple event-input model, a
   static-physical-input model, and a combined model. Begin with regularized
   logistic regression and a modest boosted-tree model. Keep the old FEMA
   classifier as a separate comparison; do not turn its outputs into truth.
4. Use a proper binary log loss (or a likelihood matched to the sampling
   design) for `flooded` versus `verified_dry`. Handle class imbalance through
   recorded sampling weights, not by changing unknown rows into negatives.
   Fit imputation, scaling, calibration, and thresholds on training/validation
   data only.
5. Hold out **whole events** to test new-storm performance and **whole
   geographic blocks or watersheds** to test transfer to new places. Keep an
   additional test slice for FEMA-minimal and no-match locations where observed
   outcomes exist. Do not claim performance in unmapped areas if that slice has
   no verified labels.
6. Report event-wise and location-wise precision, recall, false alarms, missed
   floods, PR AUC, Brier score, reliability/calibration, and coverage. Compare
   against the simple baselines. Inspect coastal and riverine cases separately.
   Choose any alert threshold on validation data according to the intended use.
7. Export a versioned model together with its target definition, training
   events, feature sources, geographic coverage, missing-input rules, and test
   results. Show a score only where its inputs and validated use domain apply;
   otherwise return `insufficient evidence` with the reason.

## Score shown to an unmapped address

The chosen score is **"estimated chance of outdoor inundation during this
specified event"** after calibration on independent event observations.
It must name the event conditions and model coverage. Do not display it as an
annual flood chance or as the probability a building interior floods. For a
timeless screening view, a relative susceptibility rank may be shown only if
it is validated against independent outcomes and explicitly labeled as a rank.
An annual probability requires event frequencies or longitudinal property-year
exposure plus an observation process that can distinguish no flood from no
report; it cannot be inferred by rescaling the current FEMA classifier.

## Implementation order

1. Pick a pilot subregion and assemble an event/footprint inventory.
2. Produce the first audited location-by-event table with positive, verified
   dry, and unknown outcomes.
3. Train and evaluate the baselines above, including a held-out unmapped slice.
4. Only after that evaluation, add the new score to the API and UI alongside
   the independently sourced FEMA benchmark.

No new production flood probability should be exported from the present FEMA
label table: it cannot test the outcome the project now wants to predict.

## High-elevation minimal comparison

The requested equal-size FEMA-label sample is available as
`data/fema-high-elevation-balanced-training.csv`. It contains all 1,238
`one_percent` rows in the geographic **training** partition and the 1,238
highest-elevation `minimal` rows in that partition (minimum selected elevation
282.1 m). Validation and test partitions remain unchanged. Rebuild and assess
the comparison with `npm run model:experiment:high-elevation`; the full
results are in `reports/fema-high-elevation-baseline.md`.

The comparison strongly argues against using this sample for the live score:
balanced accuracy on the unchanged geographic test set falls from 0.944 to
0.603. Every test `minimal` row below the selected elevation cutoff is called
higher-risk. This is selection bias from choosing only high-elevation negatives,
not evidence that those low-elevation locations flooded. The sample is a
FEMA-map classification experiment and supplies no event-inundation labels.

## Terrain-feature retraining result

`npm run model:experiment:terrain` retrains two separate XGBoost models with
five additional features already present in the calibration table: slope,
local elevation at 250 m and 500 m, log flow accumulation, and height above
drainage. One uses the full geographic training partition; the other uses the
equal-size high-elevation minimal sample above. Both keep the original
validation and test partitions. Their joblib files are under `models/`, and
`reports/terrain-fema-baselines.md` records the comparison.

On the geographic test set, the full-terrain model has 0.935 balanced accuracy,
0.994 ROC AUC, and 0.026 Brier score; the current app model has 0.944, 0.987,
and 0.032 respectively. The high-elevation terrain variant has 0.610 balanced
accuracy. The full-terrain model changes the error tradeoff but does not
improve the chosen balanced-accuracy measure, and neither variant is tested
against observed storm inundation. They remain research artifacts. The app
continues to use the original five-input model.
