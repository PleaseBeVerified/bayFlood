# Terrain training coverage audit

Counts use the shared 1,920 m CNN geographic buffer. No external rows enter training.

| Class | Training points | Flat, HAND <3 m | Elevation ≥50 m | Elevated river valley |
| --- | ---: | ---: | ---: | ---: |
| minimal | 4037 | 202 | 3153 | 1151 |
| one_percent | 945 | 625 | 261 | 45 |
| point_two_percent | 539 | 96 | 315 | 58 |

Flat = slope <1°. Elevated river valley = elevation ≥50 m, HAND ≥3 m,
and elevation relative to the 500 m neighborhood <−5 m. These are sampling strata.

| External region | Class | No matching training cell | Fewer than 10 matching rows |
| --- | --- | ---: | ---: |
| santa-cruz-north | minimal | 0/150 | 2/150 |
| santa-cruz-north | one_percent | 12/150 | 83/150 |
| santa-cruz-north | point_two_percent | 0/150 | 33/150 |
| tracy-lathrop-stockton | minimal | 0/150 | 4/150 |
| tracy-lathrop-stockton | one_percent | 0/150 | 22/150 |
| tracy-lathrop-stockton | point_two_percent | 0/150 | 12/150 |

Cells cross elevation, HAND, and slope bins listed in the adjacent JSON.

Training-only weight file: `data/terrain-coverage-training-weights.csv`.
Weights retain every training row, preserve class totals, and stay within 0.25–4.

Original calibration rows do not carry source polygon IDs or watershed IDs; a 2 km block is only a provisional clustering proxy.

Santa Cruz and San Joaquin diagnostics motivated this design; they are development diagnostics, not untouched final tests.

See [the acquisition and validation plan](../docs/terrain-generalization-training-plan.md)
and [paired weighting pilot](terrain-coverage-weighting-pilot.md).
