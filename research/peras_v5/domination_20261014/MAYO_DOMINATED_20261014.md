# Mayo diagnostics with the dominated SAA route (exploratory, aggregate only)

Cohort: all 14,195 cases / 11,447 patients of `mayo_hrs_tabular_features_and_durations.parquet`
(rows with patient and duration), note bundles from the strict pre-cutoff bundle file.  This is
NOT the 11,356-case cohort of `MAYO_CLUSTERED_FAMILY_20261014.md` (its legacy exclusions could not
be reproduced from the connected folders), so every comparison below is re-run on the same cohort.
Cluster independence and temporal stability are not attested: every number is exploratory.
No model training; no row-level output.

## Matched MAE lattice (alpha .05 / 5 nodes per route)
| Representation | Cells | Empirical SAA | Range-based cluster | Support-free block | **Dominated** |
|---|---:|---:|---:|---:|---:|
| Global | 1 | 81.246 | 67.243 | **75.841** | 70.050 |
| Procedure family | 8 | 61.120 | 47.117 | 51.887 | **51.949** |
| Procedure | 163 | 45.148 | 31.145 | 30.673 | **37.448** |
| Procedure + note metadata | 364 | 44.342 | 30.339 | 26.443 | **36.713** |
| Procedure + hashed bag | 602 | 43.712 | 29.709 | 24.637 | **36.140** |

Fragmented note representations no longer collapse: the dominated route concentrates over all
11,447 patients regardless of cell count; only within-cell optimism remains.  Global loses to the
block route because the six-case maximum cluster inflates the Bentkus common range; the new
`retention_grid` option (e.g. first two cases per patient) addresses this (procedure: 38.74 with
retention 2 at alpha .01, exploratory).

## Multi-loss lower endpoints (dominated, alpha .01 per node and loss)
| Representation | MAE | Q90 pinball | W30 miss (|error| > 30 min) |
|---|---:|---:|---:|
| Global | 70.05 | 18.12 | 0.676 |
| Family | 51.95 | 13.70 | 0.551 |
| Procedure | 37.45 | 9.93 | 0.408 |
| Procedure + metadata | 36.71 | 9.64 | 0.395 |
| Procedure + hashed bag | 36.14 | 9.36 | 0.384 |

## Fixed-policy information value on the 2,839-case temporal holdout
Coarse lower (dominated, alpha .025) minus the saved 31.5564 policy's dKelly upper endpoint
33.8533 (alpha .025, from `mayo_fixed_policy_value_31_5564_exploratory.json`):

| Coarse information | Cells | Coarse lower | Value lower bound |
|---|---:|---:|---:|
| Procedure family (regex, 8) | 8 | 43.743 | **9.890 min** |
| Procedure | 112 | 30.843 | 0.000 |

Previously both were 0 (coarse lowers 30.23 / 17.69).  Reading (exploratory): any predictor
using the policy's pre-outcome information beats the best family-only rule by at least ~9.9 MAE
minutes on this holdout.  Caveats: the holdout was viewed during development; the family map here
is the 8-group regex, not the 10-family column of the policy file; the two coarse nodes were each
run at alpha .025 (a joint ledger would use .0125 each).
