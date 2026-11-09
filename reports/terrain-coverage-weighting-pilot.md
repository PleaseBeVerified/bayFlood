# Training coverage weighting pilot

Prespecified mild weighting; same rows, features, architecture, seed and preprocessing within pairs. Binary thresholds target 95% Bay validation recall. Three-class control and coverage both use class-balanced loss. DNN training uses a 500-iteration maximum and training-loss convergence; no candidate chosen on external scores.

Each cell is mean balanced accuracy across three paired seeds: control → coverage weighting.

| Model | Bay test | Santa Cruz | Tracy–Lathrop–Stockton |
| --- | ---: | ---: | ---: |
| binary_five_input_boosted | 0.793 → 0.799 | 0.763 → 0.699 | 0.583 → 0.596 |
| binary_terrain_boosted | 0.913 → 0.915 | 0.822 → 0.961 | 0.597 → 0.597 |
| binary_five_input_dnn | 0.829 → 0.800 | 0.674 → 0.726 | 0.626 → 0.588 |
| binary_terrain_dnn | 0.911 → 0.901 | 0.706 → 0.693 | 0.652 → 0.591 |
| three_class_five_input_dnn | 0.731 → 0.708 | 0.475 → 0.443 | 0.506 → 0.536 |
| three_class_terrain_boosted | 0.763 → 0.764 | 0.545 → 0.613 | 0.510 → 0.536 |
| three_class_terrain_dnn | 0.730 → 0.723 | 0.420 → 0.450 | 0.534 → 0.517 |

Three-class 0.2% recall must also be inspected; higher balanced accuracy
can coexist with abandoning that class:

| Model | Santa Cruz 0.2% recall | San Joaquin 0.2% recall |
| --- | ---: | ---: |
| three_class_five_input_dnn | 0.000 → 0.009 | 0.027 → 0.022 |
| three_class_terrain_boosted | 0.411 → 0.378 | 0.067 → 0.000 |
| three_class_terrain_dnn | 0.038 → 0.018 | 0.113 → 0.036 |

Runs with fit warnings: 5/42. Inspect the JSON for convergence warnings before treating DNN differences as stable.

Pilot configurations are not exact retrains of every saved checkpoint. CNNs were not retrained. Three seeds measure initialization variation, not independent regional uncertainty. Existing external results motivated the design; a fresh geographic test is required. No production model replaced.
