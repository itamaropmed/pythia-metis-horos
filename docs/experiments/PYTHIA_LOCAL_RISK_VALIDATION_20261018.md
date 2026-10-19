# Pythia support-aware local risk: development results

## Synthetic clustered audit

The frozen design uses 400 independent patients, two perfectly dependent rows
per patient, two cells, and two frozen models. Known cell and global W30 risks
are available analytically. Across 100 development Monte Carlo repetitions:

| result | value |
|---|---:|
| simultaneous family coverage | 100/100 |
| false certified local directions | 0 |
| certified higher-than-global routes | 103 |
| certified lower-than-global routes | 150 |
| unresolved local routes | 147 |
| median interval width | 0.1782 |

This is a development audit, not confirmatory evidence. The important stress is
perfect within-patient dependence: Pythia clusters the repeated rows rather
than claiming 800 independent observations.

Frozen aggregate JSON:
`docs/results/local_model_risk_validation_20261018.json`.

## Mayo retrospective replay

Two honest model arms were reconstructed on the existing patient-grouped
temporal split: basic XGBoost and XGBoost plus the 11 pre-cutoff LLM features.
The outcome-blind hierarchy uses predeclared procedure family, development-fit
prediction bins, disagreement bins, and deterministic backoff. Pythia consumes
the development OOF W30 losses and trains no model.

| measure | basic | +11 LLM |
|---|---:|---:|
| development rows | 9,966 | 9,966 |
| later rows | 2,599 | 2,599 |
| development/later patient overlap | 0 | 0 |
| later observed W30 miss | 0.3817 | 0.3786 |
| certified higher/lower cases | 1,150 | 1,120 |
| unresolved local-shift cases | 1,449 | 1,479 |
| median simultaneous interval width | 0.2391 | 0.2354 |

Every later case reached at least 100 independent development patients through
the hierarchy; none required the global-only node. Most used the finest
family/prediction/disagreement cell (1,986/2,599).

The later cohort is a transport diagnostic, not a new coverage experiment. Of
the 20 routed cells with at least 50 later cases, the raw later cell miss rate
fell inside the development interval in 19/20 basic-arm cells and 18/20 +11
cells. This descriptive comparison ignores later-cohort sampling uncertainty
and must not be read as nominal 95% temporal coverage. Among certified-direction
cells with at least 20 later rows, 20/21 directions agreed with the later-vs-
global ordering for each model.

Aggregate output (outside Git):
`/Users/itamarzernitsky/Documents/clinical notes/output/pythia_mayo_local_w30/summary.json`.

## Honest interpretation

This improves Pythia by creating a valid local *frozen-model risk* object, not
by optimizing toward a correlation threshold. It does not replace the global
information-floor bound, prove individual predictability, or establish
prospective transport. The intervals are still wide; hierarchical pooling and
shift-aware transport are the next sharpness problems.

---
**Update 2026-10-18.** The OOF-based Mayo route reported here is superseded for confirmatory claims by the v2 route (split calibration + support partition + contrast). See `docs/experiments/PYTHIA_MAYO_LOCAL_W30_V2_20261018.md`. On a Mayo plasmode with known truth, the v1 route gave wrong certified directions in 24/25 replicates; v2 gave none.
