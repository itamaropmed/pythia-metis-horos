# Mayo local W30 certificate: v2 route and plasmode validation (run locally, 2026-10-18)

All computation ran on the owner's machine, in the Cowork VM, on the registered working set. The `df.parquet`, `XA.parquet` and `XB.parquet` SHA-256 hashes match `docs/MAYO_DATA_MANIFEST.json`.

- Only aggregates were written to the repository: `research/pythia_local_20261018/mayo_local/`. Every reported region has at least 52 later cases.
- No identifiers or note text were written anywhere.

**Scripts**
- `scripts/run_mayo_pythia_local_w30_v2.py`: the real-outcome replay.
- `scripts/run_mayo_pythia_local_w30_plasmode.py`: validation with a known law on real Mayo covariates.

**Theory and claims:** `docs/PYTHIA_LOCAL_RISK_CLAIMS_BACKBONE_20261018.md`.

## 1. Real Mayo outcomes: v2 route vs the original v1 route

**Split.** Development is the period before the 2025-08-25 cutoff. It contains 4,939 FIT patients (6,244 cases) and 3,292 CALIBRATION patients (3,722 cases); the two groups do not overlap. The later cohort has 2,301 patients (2,599 cases), with no patient overlap with development.

**Declared before outcomes:** m = 100 (primary), α = 0.05, fit share 0.6.

| | Basic XGBoost | XGBoost + 11 LLM |
|---|---|---|
| **v1 (OOF route), certified higher / lower / unresolved** | 556 / 603 / 1,440 (44.6% certified) | 556 / 603 / 1,440 (44.6%) |
| **v2 primary (m = 100), certified higher / lower / unresolved** | **814 / 604 / 1,181 (54.6%)** | **814 / 446 / 1,339 (48.5%)** |
| v2 regions, median interval width | 22 regions, 0.300 | 22 regions, 0.299 |
| Certified regions whose later observed rate agrees in direction | **12 / 12** | **11 / 11** |
| Later observed W30 inside the development interval (regions with ≥ 50 later cases): Λ = 1 / 1.25 / 1.5 | 20 / 21 / 21 of 22 | 20 / 21 / 21 of 22 |
| Median tipping Λ\* of certified directions | 1.55 | 1.42 |
| Harmful-shift monitor alarms | 0 | 0 |
| Later MAE (v2 model fitted on 60% of development) | 33.25 min | 32.45 min |
| Later W30 | 38.4% | 37.2% |
| Sensitivity, m = 200 (predeclared, not selectable): certified share | 66.2% | 56.5% |
| Sensitivity, m = 400 (predeclared, not selectable): certified share | 73.1% | 73.1% |

**Reading**
- **v2 is valid by construction and still certifies more cases than v1** (54.6% / 48.5% against 44.6%).
- Every certified direction agreed with the later cohort.
- The certified directions are robust to about 40–55% localized odds drift.
- **The cost is accuracy.** The deployed v2 model is fitted on 60% of development patients, which adds about 1.0 / 0.5 minutes of MAE (v1's refit had 32.23 / 31.92).
- **This replay recomputed v1 and got 1,159 certified cases for both arms. The earlier report gave 1,150 / 1,120.** The difference comes from the XGBoost seed and thread settings in this run.

## 2. Plasmode validation on the real Mayo design (known truth)

**What is kept from Mayo:** covariates, families, patients and repeated cases, scheduled times and the temporal split.

**What is simulated:** only the outcome, as y\* = m(x) + √0.7·r + u_patient, where
- m is a fixed XGBoost mean fitted on all real rows;
- r is drawn from the real residual pool of the case's procedure family;
- u is a shared patient effect carrying 30% of the family variance.

The true W30 risk of every frozen model on every case is then computed exactly. Over 25 replicates, with both models in each:

| Route | Stated-estimand family coverage | Truth of routed cases outside their interval | **Wrong certified directions** | Certified share |
|---|---|---|---|---|
| **v1** (OOF route, later cohort) | 24/25 | 172 cell misses | **1,476 / 21,733 rows (6.8%); 24/25 replicates** | 16.7% |
| **v2**, calibration population (the theorem's estimand) | **25/25** [CP 95%: 0.86, 1] | 0 | **0 / 21,345** | 11.5% |
| **v2**, transported to the later cohort (real covariate drift, same outcome law) | 25/25 | 0 | **0 / 15,637** | 12.0% |

Later MAE: v1 32.71 / 32.09; v2 33.39 / 32.74.

**Outcome-law drift.** Later residuals were inflated by 30% over 12 replicates. Calibration-population coverage stayed exact: 0 misses and 0 wrong directions. For the later cohort:

| Λ | Later truth inside the transported interval | Replicates fully covered |
|---|---|---|
| 1.0 | 0.912 | 0 / 12 |
| 1.25 | 0.996 | 10 / 12 |
| 1.5 | 1.000 | 12 / 12 |

There were no wrong directions: 0 of 7,252 certified rows.

## Conclusions for Mayo

1. **Replace the v1 route.** On the real Mayo design it produced wrong certified directions in 24 of 25 replicates with a known law. v2 produced none in 37 replicates, including drift.
2. **Claims allowed:**
   - the v2 directions are `conditional` claims for the calibration population of each routed region;
   - later-cohort use is `conditional on Λ`, reported together with Λ\* (median about 1.4–1.6 on real data);
   - a declared Λ of 1.25–1.5 restored coverage under 30% residual inflation.
3. **The price is about 0.5–1.0 minute of MAE from fitting on 60%.** Next experiment: a predeclared fit share of 0.8, which gives a smaller calibration set and the same theorem, or a proved cross-fitted variant.
4. **Still retrospective.** Prospective confirmation follows `docs/PROSPECTIVE_PROTOCOL.md`.
