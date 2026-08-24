# FEMA area-rate pipeline

Status: **AREA_SIGNAL_READY__PARCEL_PROBABILITY_BLOCKED**

Complete Bay Area v3 records were aggregated for 2009-2025.

- Claims: 2782
- Policy-years: 788228.88
- Census block groups: 4659
- Area-years: 54484

| Split | Area-years | Policy-years | Observed claims | Expected claims | O/E |
|---|---:|---:|---:|---:|---:|
| test | 9347 | 117014.75 | 367 | 473.47 | 0.7751 |
| train | 32625 | 514081.73 | 1912 | 777.28 | 2.4598 |
| validation | 9723 | 130254.89 | 449 | 503.76 | 0.8913 |

The lagged signal uses only prior years. It is suitable as an area-level model feature, not as a property flood probability.
