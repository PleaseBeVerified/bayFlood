# Binary CNN false positives: Tracy–Lathrop–Stockton

The saved CNN flags 104/150 FEMA mapped minimal points (69.3%) at its original Bay Area validation threshold (0.06183).
All 150 mapped 1% points are detected. No model or threshold was changed.

## Findings

The main descriptive pattern is low, flat terrain with low height above drainage,
both at the sampled point and throughout its roughly 1.92 km wide image patch.
Water proximity and flow accumulation overlap substantially between flagged and cleared
minimal points, so those inputs alone give a weaker explanation.

Two polygons contribute 92 of 104 false positives, but also contain 135 of 150
minimal samples. The terrain pattern persists within polygon 28049960:
flagged versus cleared median elevation is 13.1 versus 38.7 m, and height
above drainage is 2.5 versus 8.7 m. Spatial correlation still limits inference.

Complete raster coverage and the boundary audit provide no evidence that these
errors are mainly missing pixels or tiny coordinate shifts across FEMA boundaries.

The recall-oriented threshold amplifies the false positives. Raising the score
threshold to 0.5 would still leave 42 false positives and reduce detection of
mapped 1% points from 150 to 107. Threshold sensitivity does not justify
selecting a new operating point on this external test set.

## Terrain pattern

Medians, with Q25–Q75 in parentheses:

| Input | Flagged minimal | Correctly cleared minimal |
| --- | ---: | ---: |
| elevation_m | 13.041 (10.528–15.304) | 37.982 (24.523–63.008) |
| height_above_drainage_m | 1.184 (0.613–2.200) | 7.518 (3.767–12.175) |
| slope_degrees | 0.277 (0.109–0.585) | 1.034 (0.647–1.314) |
| nearest_water_distance_m | 324.000 (96.250–595.250) | 427.000 (167.000–621.500) |
| current_imperviousness_pct | 0.000 (0.000–52.000) | 18.000 (0.000–58.000) |
| current_100yr_24h_rainfall_mm | 86.411 (78.384–87.719) | 74.650 (70.796–78.181) |
| local_elevation_250m | 0.018 (-0.137–0.190) | -0.021 (-0.322–0.159) |
| local_elevation_500m | 0.009 (-0.171–0.190) | -0.056 (-0.595–0.379) |
| flow_accumulation_m2 | 1500.000 (475.000–10925.000) | 2900.000 (800.000–16950.000) |
| patch_median_slope_degrees | 0.255 (0.166–0.596) | 0.861 (0.581–1.509) |
| patch_median_height_above_drainage_m | 1.105 (0.807–1.643) | 6.205 (3.735–11.259) |
| patch_dem_iqr_m | 1.238 (1.023–3.150) | 9.059 (4.567–13.717) |
| cnn_score | 0.386 (0.152–0.782) | 0.012 (0.001–0.027) |
| fema_polygon_boundary_distance_m | 1626.303 (434.117–2895.725) | 3128.801 (1995.768–4023.009) |

## Spatial concentration

| FEMA polygon | Minimal points | False positives | Rate |
| --- | ---: | ---: | ---: |
| 28049960 | 75 | 34 | 45.3% |
| 28049400 | 60 | 58 | 96.7% |
| 28048826 | 4 | 4 | 100.0% |
| 28048942 | 2 | 1 | 50.0% |
| 28048939 | 2 | 2 | 100.0% |
| 28049777 | 1 | 0 | 0.0% |
| 28049062 | 1 | 1 | 100.0% |
| 28049050 | 1 | 1 | 100.0% |
| 28049272 | 1 | 1 | 100.0% |
| 28049848 | 1 | 0 | 0.0% |
| 28048926 | 1 | 1 | 100.0% |
| 28049918 | 1 | 1 | 100.0% |

Concentration must be interpreted against the number of sampled points per polygon.
Polygon group medians and Bay Area test feature comparisons are in the adjacent JSON.

## Bay Area input contrast

Medians in the saved Bay Area test set versus the new region's false positives:

| Feature | Bay minimal (812) | Bay mapped 1% (342) | New-region false positives (104) |
| --- | ---: | ---: | ---: |
| elevation_m | 127.600 | 1.000 | 13.041 |
| height_above_drainage_m | 41.303 | 0.011 | 1.184 |
| slope_degrees | 11.170 | 0.262 | 0.277 |

This is consistent with geographic transfer failure: terrain resembling the Bay
lowland positive class can still have a minimal FEMA label in San Joaquin County.
This association does not establish which CNN inputs cause an individual prediction.

## Terrain strata

| Feature | Range | False positives / minimal points |
| --- | --- | ---: |
| elevation_m | [-inf, 10) | 22/27 |
| elevation_m | [10, 20) | 72/75 |
| elevation_m | [20, 40) | 9/27 |
| elevation_m | [40, inf) | 1/21 |
| height_above_drainage_m | [-inf, 1) | 46/49 |
| height_above_drainage_m | [1, 3) | 45/50 |
| height_above_drainage_m | [3, 10) | 11/32 |
| height_above_drainage_m | [10, inf) | 2/19 |

## Score and threshold diagnostic

The original threshold was selected for at least 95% recall on Bay Area validation.
The threshold alternatives below are descriptive; they were not applied to the model.

| Score threshold | Minimal false positives / 150 | Mapped 1% detected / 150 |
| --- | ---: | ---: |
| 0.06183 | 104 | 150 |
| 0.10000 | 91 | 150 |
| 0.20000 | 70 | 147 |
| 0.50000 | 42 | 107 |
| 0.80000 | 24 | 44 |

## Data checks and limits

```json
{
  "allMinimalPointsInsideRecordedSourcePolygon": true,
  "allMinimalLabelsExplicitFemaUnshadedZoneX": true,
  "minimumTopographyCoverage": 1.0,
  "minimumImperviousnessCoverage": 1.0,
  "falsePositivesWithin30mOfFemaBoundary": 1,
  "falsePositivesWithin100mOfFemaBoundary": 7,
  "falsePositivesMoreThan1400mInsideFemaPolygon": 57
}
```

Descriptive spatially correlated sample; associations do not isolate causal model drivers. Scores measure mapped-class agreement, not event probabilities. Threshold alternatives use test labels for diagnosis only and must not be selected for deployment from this analysis.

Per-point audit: adjacent CSV. Reproduce with `python3 scripts/analyze-tracy-lathrop-stockton-cnn-errors.py`.
