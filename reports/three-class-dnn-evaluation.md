# Three-Class DNN Evaluation

This candidate predicts FEMA-equivalent mapped hazard categories from five
physical inputs. The 0.2%-annual-chance category is included. Training and
selection used geographically separated training and validation blocks;
the test blocks were evaluated only after selecting the model.

## Data

- Training rows: 7364
- Validation rows: 1336
- Test rows: 1300
- Hidden layers: [64, 32, 16]
- Features: elevation_m, current_imperviousness_pct, current_100yr_24h_rainfall_mm, nearest_water_available, log_nearest_water_distance_m

## Held-Out Test Results

- Accuracy: 0.778
- Balanced accuracy: 0.787
- Macro F1: 0.723
- Log loss: 0.542
- Majority-class baseline accuracy: 0.625

| FEMA-equivalent class | Test rows | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| minimal | 812 | 0.944 | 0.752 | 0.838 |
| one_percent | 342 | 0.768 | 0.842 | 0.803 |
| point_two_percent | 146 | 0.403 | 0.767 | 0.528 |

Confusion matrix (rows are actual, columns predicted; class order:
`minimal, one_percent, point_two_percent`): `[[611, 77, 124], [12, 288, 42], [24, 10, 112]]`.

## Newark Spot Check

Within 3 km of Newark City Hall, 10 held-out test rows were
available. The model correctly classified 2 of them;
8 mapped minimal-hazard rows were predicted
as the 0.2%-annual-chance class. This post-hoc diagnostic was not used to
select the model. It shows why this candidate is not ready to fill map gaps.

These are results for predicting FEMA map classes in held-out Bay Area blocks.
They do not validate performance in genuinely unmapped areas, and class
probabilities are not the chance that a property will flood. The model omits
local drainage, levees, and building floor elevation. Review errors and
uncertainty before displaying this candidate in the app.
