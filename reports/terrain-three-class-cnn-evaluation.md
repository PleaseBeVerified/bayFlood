# Three-class terrain CNN experiment

The CNN uses 64 × 64 terrain and imperviousness patches plus ten point inputs.
All 933 mapped 0.2% locations are eligible. A 1,920 m buffer removes training
and validation patches that could overlap a held-out patch. The boosted
comparator uses the same buffered locations and ten point inputs.

Buffered rows: training 5521, validation 1166, test 1300.
Selected CNN: balanced class weight, epoch 5, chosen on validation only.

| Geographic test metric | CNN | Buffered terrain boosted |
| --- | ---: | ---: |
| Accuracy | 0.817 | 0.832 |
| Balanced accuracy | 0.768 | 0.765 |
| Macro F1 | 0.742 | 0.749 |
| Log loss | 0.467 | 0.426 |

| Class | Test samples | CNN precision | CNN recall | CNN F1 | Boosted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| minimal | 812 | 0.909 | 0.826 | 0.866 | 0.873 |
| one_percent | 342 | 0.872 | 0.895 | 0.883 | 0.894 |
| point_two_percent | 146 | 0.403 | 0.582 | 0.476 | 0.481 |

CNN confusion matrix (actual rows, predicted columns; class order minimal, one_percent, point_two_percent): `[[671, 38, 103], [13, 306, 23], [54, 7, 85]]`.
Boosted confusion matrix: `[[685, 47, 80], [10, 321, 11], [63, 8, 75]]`.

These scores measure agreement with FEMA map classes, not observed floods.
The test blocks have been used in earlier model comparisons, so this is
a diagnostic comparison rather than a fresh independent confirmation.
Levee-related or otherwise unavailable map categories are not labels here.
