# Training data redesign for geographic generalization

The priority is coverage of **FEMA class × terrain setting × hydrologic regime × geography**.
Class balancing or adding more random points from the current polygons is insufficient.
The audit and paired pilots support a new collection and validation protocol; they do
not establish one weighting recipe that improves every model.

## Measured coverage gaps

These counts use the same 1,920 m geographic exclusion buffer as the three-class
CNN experiment: 5,521 training, 1,166 validation, and 1,300 Bay test rows. Binary
experiments exclude the 0.2% class. Terrain thresholds are explicit descriptive
strata, not physical flood boundaries.

| Class and terrain setting | Buffered Bay training | Relevant diagnostic region |
| --- | ---: | ---: |
| Minimal: slope <1°, height above drainage <3 m | 202/4,037 (5.0%) | Tracy–Lathrop–Stockton: 87/150 (58.0%) |
| Mapped 1%: elevation ≥50 m, height above drainage ≥3 m, local elevation at 500 m <−5 m | 45/945 (4.8%) | Santa Cruz: 62/150 (41.3%) |
| Mapped 0.2%: same elevated river-valley setting | 58/539 (10.8%) | Santa Cruz: 113/150 (75.3%) |
| Mapped 0.2%: slope <1°, height above drainage <3 m | 96/539 (17.8%) | Tracy–Lathrop–Stockton: 129/150 (86.0%) |

Twelve of 150 Santa Cruz 1% points occupy same-class elevation/HAND/slope cells
with **zero** buffered training examples; 83/150 have fewer than ten training
examples in their cell. Conversely, all 150 San Joaquin minimal points have
some same-class cell support, with only four below ten examples. The San Joaquin
problem therefore includes a change in geographic context and label separation,
rather than complete absence of those coarse terrain cells.

Cells use elevation edges 0, 5, 20, 50, 150 m; height-above-drainage edges 1, 3,
10, 30 m; and slope edges 1, 5, 15°. These are a coverage dashboard. Filling
every possible Cartesian cell is neither feasible nor desirable: some settings
do not occur in a given class.

## Collection priorities

The following are proposed starting acquisition goals, not estimated optimal
sample sizes. Count new independent locations and spatial groups, not duplicated
rows. Check source feasibility before enforcing quotas.

| Priority | New examples to acquire | Proposed first tranche | Why |
| --- | --- | --- | --- |
| 1 | Explicit FEMA minimal areas on flat inland plains, including rural canals and urban drainage settings | About 300 additional points across ≥3 training watersheds or regions | Counter low-flat terrain being treated as a near-universal 1% signature |
| 1 | Explicit mapped 1% areas in elevated river valleys and terraces, including larger HAND values | About 300 points across ≥3 training watersheds | Address Santa Cruz-like terrain absent or sparse in Bay positives |
| 1 | Mapped 0.2% flat inland settings **and** elevated river-valley settings | About 300 points in each setting, spread across multiple source polygons and watersheds | The 0.2% class is both sparse and geographically shifted |
| 2 | Minimal, 1%, and 0.2% locations with similar terrain and land cover in the same hydrologic regime | Matched groups inside each acquired watershed | Expose which additional predictors distinguish classes instead of reinforcing elevation shortcuts |
| 2 | Coastal lowlands, inland alluvial plains, incised valleys, foothills, steep headwaters; rural, mixed, and urban land cover | Coverage floors based on available independent polygons/blocks | Retain easy examples and prevent the next rare-regime failure |

Retain existing high-elevation minimal and low-elevation positive examples. The
earlier high-elevation-negative-only experiment removed the difficult lowland
negatives and flagged all 300 San Joaquin binary points. It is not a suitable
generalization strategy.

Acquire the new points in designated training regions. Keep the current 900
Santa Cruz and San Joaquin rows out of training for this investigation. These
regions have already informed the redesign, so their scores are **development
diagnostics**. A new county/watershed, such as a separately prepared western
Stanislaus region, should be frozen before any tuning and reserved for final
testing. Do not use the same new region for both collection and final validation.

## Sampling and provenance rules

1. Allocate by class, terrain setting, hydrologic regime, and region. Within
   each allocation, spread samples among source polygons and spatial blocks;
   do not let a large polygon dominate because of its area alone. Use quotas
   or group weights and record infeasible cells rather than inventing labels.
2. Add `source_object_id`, DFIRM/map ID, raw zone and subtype, effective date,
   source URL/hash, watershed ID, spatial block, hydrologic regime, and water
   source/type to the master table. The original calibration table lacks source
   polygon and watershed IDs, so a 2 km block is currently only a clustering proxy.
3. Keep explicit minimal, 1%, 0.2%, levee-related, unmapped, and uncertain labels
   distinguishable. Do not turn unmapped or levee-related Zone X into minimal,
   or assign 0.2% rows a binary minimal target. These remain map-class models;
   observed storm models need event-specific flood and verified-dry evidence.
4. Recompute GIS predictors using consistent source definitions and vintages.
   The new regions use supplemental 3DHP water data while the original training
   values came from older local hydrography. Audit water distances, canals versus
   rivers, HAND/network definitions, rainfall units, and raster validity before
   interpreting a source mismatch as a learnable terrain pattern.
5. Split by watershed or large geographic blocks **before** augmentation or
   hard-example mining. Keep ≥1,920 m Chebyshev separation between CNN training
   centers and validation/test centers for the current 64×64, 30 m patches.
   Retain group IDs so overlapping patches and correlated polygon samples do
   not masquerade as independent validation.

## A common data interface for all models

The investigation writes `data/terrain-coverage-training-weights.csv`. It includes
only the 5,521 buffered training IDs, with separate binary and three-class
coverage weights. Weights use inverse-square-root terrain-cell, county, and
within-cell spatial-block density, normalized within each class and capped at
0.25–4. They preserve class totals and retain every training row. Binary weights
are recomputed after removing 0.2% rows; CNN weights must likewise be recomputed
if its actual training subset changes. Class balancing is an additional,
separate loss choice, not a substitute for terrain coverage.

The weighting reduces effective sample size to approximately 2,986 three-class
or 2,754 binary observations. It cannot create missing terrain, hydrologic
mechanisms, or independent spatial groups. The file is an experimental input,
not a production replacement dataset.

| Family | Apply the common sampling design |
| --- | --- |
| Five-input boosted/DNN | Use the same IDs and terrain-derived sampling weights even when only five predictors enter the model; assess whether the limited feature set can resolve matched terrain classes |
| Ten-input boosted/DNN | Use `sample_weight`; fit preprocessing on the training partition only and select checkpoints/thresholds on geographic validation |
| Binary and three-class CNN | Use the same sample IDs and weighted per-example loss or weighted minibatch sampling; avoid applying both without accounting for the doubled correction; retain aligned channel masks |

The existing binary CNN already tested a 1.5× loss weight for minimal points
below 50 m, but Bay validation selected the unweighted candidate. An elevation
cutoff alone is a weak proxy for the class × terrain × regime × geography
coverage problem. Rotations and flips may regularize images; they do not add
new terrain regimes or correct missing label distinctions.

The shared weighting pilot covers seven tabular family/target combinations,
each with three paired seeds and the same buffered data. Results are in
[`terrain-coverage-weighting-pilot.md`](../reports/terrain-coverage-weighting-pilot.md).
The separate CNN experiment below uses the same weighting recipe and buffered
IDs. Both pilots measure reweighting existing examples; neither evaluates the
proposed new acquisition tranche.

The 42 fits (seven configurations × three seeds × control/weighted) produced
these representative mean balanced accuracies:

| Paired pilot | Santa Cruz, control → weighted | San Joaquin, control → weighted |
| --- | ---: | ---: |
| Binary terrain boosted | 0.822 → 0.961 | 0.597 → 0.597 |
| Binary terrain DNN | 0.706 → 0.693 | 0.652 → 0.591 |
| Three-class terrain boosted | 0.545 → 0.613 | 0.510 → 0.536 |

The three-class boosted model's San Joaquin 0.2% recall nevertheless fell from
6.7% to **0%**. A better aggregate score therefore does not establish a better
three-class model. Five of 42 fits still emitted convergence warnings at the
500-iteration DNN budget. The paired controls use shared buffers and validation
rules, so these are not comparisons against the earlier saved-model tables.
Do not promote this weighting recipe as a universal improvement. Treat it as
a reproducible acquisition-baseline experiment; prioritize new cross-regime
examples and evaluate class-specific tradeoffs before retraining the full suite.

## Paired CNN results

The [CNN coverage pilot](../reports/terrain-cnn-coverage-weighting-pilot.md)
completed 12 fits: binary and three-class targets × control/coverage loss ×
three seeds. Binary training uses 14 epochs; three-class training uses 12.
Each pair shares its initial seed, minibatch order, training-only preprocessing,
buffered IDs, architecture, and epoch budget. Checkpoints are selected only on
Bay geographic validation. Three-class loss is class balanced in both variants;
the additional coverage factor is the experimental change. External regions
are scored after checkpoint selection. These paired controls differ from the
earlier saved app checkpoints.

| CNN | Bay test balanced accuracy | Santa Cruz | San Joaquin |
| --- | ---: | ---: | ---: |
| Binary, control → coverage | 0.937 → 0.929 | 0.839 → 0.842 | 0.643 → 0.611 |
| Three-class, control → coverage | 0.759 → 0.745 | 0.575 → 0.594 | 0.456 → 0.475 |

The binary Santa Cruz result varies substantially by seed: control balanced
accuracy spans 0.747–0.900, and coverage spans 0.663–0.947. Its small mean gain
does not establish a stable improvement. On San Joaquin, binary minimal
false-positive rate worsens from 70.9% to 77.8%; mapped 1% recall changes from
99.6% to 100%.

Three-class Santa Cruz 0.2% recall improves from 15.8% to 21.8%, while San
Joaquin 0.2% recall is essentially unchanged (30.7% to 30.4%). Three-class
minimal error rates improve in both external regions, accompanied by lower
mapped 1% recall. Bay test balanced accuracy declines for both CNN targets.

These results reinforce the acquisition priority: modest loss reweighting
does not supply independent examples of missing contexts, and there is no
consistent gain across models and regions. Keep the recipe experimental.
CNN checkpoints remain in the experiment cache; the app models are unchanged.
Seed variation measures initialization sensitivity, not uncertainty across
new geographic regions. The GPU runs are seeded but not guaranteed bitwise
deterministic.

## Validation and acceptance

Use multiple training watersheds with geographic development folds, and select
on mean **and worst-region** performance. A pooled Bay-only score can prefer a
model that fails an inland plain or river valley. Choose binary operating
thresholds on development folds under an explicit recall/false-positive policy;
do not repair an external test by tuning its threshold.

Report per-region and per-terrain confusion matrices, minimal false-positive
rates, mapped 1% recall, 0.2% precision/recall/F1, macro F1, and ranking metrics.
Use polygon/watershed/block resampling for uncertainty. Three random seeds
measure training variability, not confidence across unseen regions. Track
source coverage and support flags at inference for unfamiliar terrain regimes.

Before promotion, require improved performance on the priority settings without
material regression on retained terrain groups, and evaluate once on the fresh
frozen geographic test. More examples cannot guarantee separation when the
available predictors omit relevant hydraulic or engineered-protection context.
Investigate additional water-type, connectivity, coastal/inland, and protection
features only when they can be sourced consistently for training and inference.

Spatial separation follows the principle discussed by
[Roberts et al., cross-validation for structured data](https://doi.org/10.1111/ecog.02881).
The tabular implementation uses the documented
[MLP sample-weight interface](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
