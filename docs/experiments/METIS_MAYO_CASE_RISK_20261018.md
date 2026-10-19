# Metis Mayo new-case risk proof of concept

Date: 2026-10-18

## Question and boundary

Can the Pythia/Metis/Horos stack receive a genuinely later Mayo HRS case and,
before observing its duration, return a numerical and categorical reliability
assessment for two frozen duration models?

This experiment separates three objects that must not be conflated:

1. **Pythia information-set floor**: a cohort-level lower bound on the MAE
   attainable from a declared representation.
2. **Horos duration prediction**: a fitted model's prediction in minutes.
3. **Metis case-risk forecast**: an OOF-calibrated estimate of how wrong that
   particular frozen model may be on the arriving case.

The Metis value is not an individual Bayes-risk lower bound. It is a predictive
risk estimate. Its uncertainty endpoint is conditional on the frozen routing
map, bounded error, and exchangeable independent patient entities.

## Mayo references reconciled

- Historical Arm A is tabular only: 296 encoded columns.
- Historical Arm B is the same table plus 11 LLM-extracted preoperative note
  concepts: 307 columns.
- Their frozen full-cohort calibrated Pythia floors are respectively 29.29 and
  28.84959960434628 minutes.
- The separate 31.556418 result is a later Foundation/PLS/meta-gate system. It
  explicitly excludes the old 11-feature table and therefore is not one of the
  two models compared here.
- The alleged 31.2623 incumbent remains unverified; the located number was a
  residual-corrector OOF metric, not a final held-out duration MAE.

The A/B replay uses the historical 12,993-row cohort, the 80th-percentile
temporal boundary, and patient exclusion across that boundary. It produces
9,966 development cases from 8,231 patients and 2,599 later cases from 2,301
new patients, with zero patient overlap.

## Protocol

Two matched XGBoost models use the same hyperparameters and randomization:

- `basic_xgboost`: Arm A;
- `xgboost_plus_11_llm`: Arm B.

Within development, five patient-grouped folds create honest OOF predictions.
Metis observes those OOF absolute errors and freezes a hierarchical risk map.
The map's keys are outcome-blind: procedure family, prediction bin, and
cross-model disagreement bin. A cell with fewer than 100 independent patients
backs off through less specific cells to a global cell.

For a new case and model, Metis returns:

- the duration prediction;
- expected absolute error in minutes (`risk_value`);
- estimated probability of error above 30 minutes;
- `low`, `moderate`, `high`, or `very_high` risk;
- support level and independent-patient count;
- conditional upper endpoints;
- the separate Pythia information-set floor and claim tier.

All 2,599 held-out risk cards were created before any held-out outcome entered
the scoring call. Outcomes were then revealed only for retrospective audit.

## Results after outcome reveal

| Model | Held-out MAE | Mean predicted MAE | P(|error| > 30), predicted | P(|error| > 30), observed | Spearman risk vs error |
|---|---:|---:|---:|---:|---:|
| basic XGBoost | 32.2312 | 32.4865 | 0.3714 | 0.3817 | **0.4385** |
| XGBoost + 11 LLM concepts | **31.9184** | 32.0485 | 0.3654 | 0.3786 | 0.4273 |

The 11-feature model gains 0.3128 MAE minutes on average, but beats the basic
model on only 50.83% of cases. The gain is therefore a distributional effect,
not evidence that the richer model is safer for every patient.

### Ordered risk groups

| Model | Risk tier | n | Predicted MAE | Observed MAE | Observed P(|error| > 30) |
|---|---|---:|---:|---:|---:|
| basic | low | 551 | 15.29 | 14.97 | 0.123 |
| basic | moderate | 1,101 | 27.31 | 26.65 | 0.334 |
| basic | high | 719 | 43.43 | 45.12 | 0.563 |
| basic | very high | 228 | 64.54 | 60.29 | 0.662 |
| +11 LLM | low | 572 | 15.32 | 15.56 | 0.121 |
| +11 LLM | moderate | 1,190 | 27.90 | 27.50 | 0.348 |
| +11 LLM | high | 603 | 44.11 | 44.87 | 0.559 |
| +11 LLM | very high | 234 | 62.94 | 60.99 | 0.701 |

The monotone separation is the main proof-of-concept result. Low-risk cases
average roughly 15 minutes of error; very-high-risk cases average 60--61
minutes. This is useful case-level discrimination, while still far from a
clinical guarantee.

This replay used the locally available XGBoost 3.2.0 runtime. Its results are
close to, but not bit-identical with, the preserved historical A/B holdout
values (32.28 and 31.67). The production gate must run the pinned environment
and hash the fitted artifacts; this POC establishes the Metis behavior, not a
new replacement headline for the historical model scores.

## Privacy and artifacts

The repository contains code and this aggregate report only. The local example
file uses salted SHA-256 case pseudonyms and contains neither patient IDs nor
note text. It remains outside Git at:

`/Users/itamarzernitsky/Documents/clinical notes/output/metis_mayo_case_risk_v1/`

Reproduction code: `scripts/run_mayo_metis_case_risk.py`. Core API:
`metis/case_risk.py`. Synthetic tests: `tests/test_metis_case_risk.py`.

## What remains before prospective use

1. Freeze an entirely new temporal lockbox; the historical held-out cohorts
   have influenced development.
2. Replace the historical floor attachment with a Pythia certificate generated
   on an evaluation contract aligned exactly to the risk-map cohort.
3. Predeclare tier thresholds and operational actions with clinicians.
4. Audit temporal drift, note-concept availability, missingness and unseen
   procedures; refuse rather than globally back off when deployment support is
   absent.
5. Tighten the currently conservative error upper endpoint using externally
   defensible procedure-local supports or a cluster-valid variance-sensitive
   theorem.
6. Evaluate calibration, selective coverage and clinical utility on the new
   lockbox without changing the map.
