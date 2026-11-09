# CNN training coverage weighting pilot

Three paired seeds per binary/three-class target. Same buffered IDs, architecture, initialization seed, minibatch order, preprocessing and epoch budget within pairs; only terrain-coverage loss weights differ. Binary: 14 epochs, Bay validation checkpoint chosen by lowest FPR at >=95% recall, then Brier. Three-class: 12 epochs with class-balanced loss in both variants; Bay validation selects macro F1, then 0.2% F1, then log loss. External regions scored only after checkpoint selection.

Completed 12/12 target/variant/seed runs.

Mean balanced accuracy across completed seeds, control → coverage weighted:

| CNN | Bay test | Santa Cruz | Tracy–Lathrop–Stockton |
| --- | ---: | ---: | ---: |
| binary | 0.937 → 0.929 | 0.839 → 0.842 | 0.643 → 0.611 |
| three_class | 0.759 → 0.745 | 0.575 → 0.594 | 0.456 → 0.475 |

Class-specific rates, control → coverage weighted:

| Target / region | Minimal false-positive rate | Mapped 1% recall | Mapped 0.2% recall |
| --- | ---: | ---: | ---: |
| binary / bay_test | 0.086 → 0.117 | 0.961 → 0.975 | — |
| binary / santa-cruz-north | 0.002 → 0.004 | 0.680 → 0.689 | — |
| binary / tracy-lathrop-stockton | 0.709 → 0.778 | 0.996 → 1.000 | — |
| three_class / bay_test | 0.211 → 0.183 | 0.889 → 0.860 | 0.598 → 0.559 |
| three_class / santa-cruz-north | 0.078 → 0.038 | 0.644 → 0.602 | 0.158 → 0.218 |
| three_class / tracy-lathrop-stockton | 0.858 → 0.789 | 0.918 → 0.909 | 0.307 → 0.304 |

Individual seed confusion matrices, score metrics, and selected epochs are in the adjacent JSON.
Class order is minimal, 1%, 0.2% (binary excludes 0.2%).

Exploratory reuse of development regions; three seeds are not independent geographic replications. MPS training is seeded but not guaranteed bitwise deterministic. These paired controls use shared buffers and are not the earlier saved checkpoints. No new terrain examples acquired and no app model replaced.

Reproduce: `.venv-model/bin/python scripts/train-terrain-cnn-coverage-pilot.py`.
Completed matching cache runs are reused; new configurations train from scratch.
