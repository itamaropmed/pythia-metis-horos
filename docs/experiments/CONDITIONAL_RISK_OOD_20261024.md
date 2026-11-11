# Conditional-risk estimator red team: ten known-truth designs

**Status:** exploratory synthetic development evidence; not a Mayo evaluation,
not a validity proof, and not a claim of statistical superiority in a broader
population of data-generating processes.

## What was evaluated

The benchmark is implemented in
[`research/conditional_risk_ood_suite.py`](../../research/conditional_risk_ood_suite.py)
with integrity tests in
[`tests/test_conditional_risk_ood_suite.py`](../../tests/test_conditional_risk_ood_suite.py).
It isolates the second-order problem: estimate the conditional expected loss
of already-frozen predictors. No base duration model is fit inside the
benchmark.

Machine-readable outputs: [10 families × 10 seeds](/private/tmp/pythia_conditional_risk_ood_10seed.json)
and [doubled-noise shift attack](/private/tmp/pythia_conditional_risk_ood_noise_shift2.json).
SHA-256: `361e020cb30e9c8bcdc101c5d32f988c27e0c1bfbe1b108d56e8f8ba9ed1299c`
and `4bb576c2390ac957020a016844f6043cf187ca0067285397c83584b5ed5bc007`,
respectively. They contain synthetic aggregates only.

Each design has 10 random seeds, 600 patients per seed, two visits per patient,
420 development patients and 180 later test patients. Forecasts and all
development-only selection/calibration are completed before held-out outcomes
are used for scoring. Patient IDs are disjoint across the split, and every
development timestamp precedes every test timestamp. The risk forecaster uses
forward-time, patient-grouped cross-fitting. The code records these controls;
the result JSON reports all 100 runs and asserts zero patient overlap and valid
temporal order.

The two frozen predictors are a conditional-mean predictor and a deliberately
biased nonlinear variant. The compared estimators are:

- Horos' temporal cross-fitted risk forecaster;
- direct squared-error regression of realized absolute loss (and separately
  of the W30 miss indicator), a standard conditional-risk-regression baseline;
- the global development mean.

All ten outcomes are generated from normal or finite normal-mixture laws. This
allows exact analytic conditional MAE and W30-probability truth. The note case
uses four synthetic pre-outcome concept indicators derived from covariates; it
is *text-like structured input*, not a clinical NLP evaluation.

## Ten-family results

Each row names the lowest-RMSE method for each frozen predictor, averaged over
the 10 seeds. The win count is therefore across 20 design × predictor cells.

| Design family | Predictor A winner | Predictor B winner |
|---|---|---|
| Homoskedastic normal | Global mean | Global mean |
| Nonlinear heteroskedastic | Horos | Horos |
| Interaction-dependent bias/noise | Horos | Horos |
| Contaminated heavy-tail mixture | Global mean | Global mean |
| Skewed noise mixture | Global mean | Horos |
| Rare high-risk subgroup | Direct regression | Direct regression |
| Conditional multimodal mixture | Horos | Horos |
| Note-concept-dependent risk | Horos | Horos |
| Patient-shared random shock | Horos | Horos |
| Covariate-shifted late test cohort | Global mean | Global mean |

| Estimator | Strict wins / 20 | Macro conditional-MAE RMSE (minutes) | Macro W30-probability RMSE |
|---|---:|---:|---:|
| Horos temporal cross-fit | 11 | 2.476 | 0.0472 |
| Direct loss regression | 2 | 3.368 | 0.0591 |
| Global mean | 7 | 3.364 | 0.0505 |

These results are mixed, not a decisive “best” result. Horos has the best
macro-average error and wins a narrow majority of the cells. It loses when the
true risk is nearly constant (where a global mean is appropriate), in the
heavy-tail mixture, and in the rare subgroup where direct regression wins.
For W30, Horos has the best macro RMSE, but the global mean wins more individual
cells (9 vs. 7; direct regression wins 4). Ten seeds improve Monte Carlo
stability within these chosen designs; they do not make the ten hand-selected
families a random sample of real-world distributions or establish a
population-level significance claim.

## Distribution-shift attack

The test-noise attack doubles outcome noise only in the later test cohort,
leaving all recorded features unchanged. Across the 20 design × predictor
cells (two seeds per design for this attack), all three estimators
underestimated true conditional MAE by about **9.6 minutes on average**. Mean
conditional-MAE RMSE rose to 11.05 minutes for Horos and 12.14 for the global
mean; W30 probability RMSE was 0.190 for Horos and 0.204 for the global mean.
This is an intentionally hidden change in `P(Y|X)`: without outcome feedback,
no method can reliably infer it from unchanged covariates alone. It motivates
drift alarms, delayed-label recalibration, and a risk-card refusal/uncertain
state; the current scores do not solve this failure mode.

## Integrity and known caveats

The test suite checks the ten designs' analytic risk ranges, patient/time
separation, text-concept signal, shared patient-shock dependence, and the
noise-doubling control. The patient-shared design now samples an actual common
latent shock across the two visits; its conditional marginal law is exactly the
normal law used by the oracle risk calculation. The direct baseline minimizes
squared error, which targets conditional mean loss; the initial absolute-loss
version was rejected as an unfair comparator and is not reported. The four
benchmark-specific tests pass, and the full repository suite now passes
**476 tests with 9 skipped** (one non-fatal legacy import deprecation warning).

The synthetic mixture outcomes have unbounded support, so this experiment does
not validate any finite-support empirical-Bernstein upper endpoint. It
evaluates point-risk error only. The rare-risk family is sparse by construction
and demonstrates a remaining support/resolution challenge. The suite has no
raw natural-language corpus, no Mayo data, no patient-cluster bootstrap
confidence intervals for a real cohort, and no preregistered or held-back
family selection.

## Research interpretation and next work

The result supports a practical, limited statement: the current Horos learner
is a competitive conditional-loss estimator on several smooth and
note-concept-driven designs, but the global baseline and direct regression
remain important comparators and can win. It does not establish novelty or
validity of a Pythia lower bound. Conditional expected-loss estimation is
already treated as a regression/calibration problem in [Vasilyev et al.,
*Calibrating Conditional Risk* (2026)](https://arxiv.org/abs/2604.20409).

The stronger research question remains a finite-sample lower confidence bound
on incremental attainable risk reduction from a declared information source,
accounting for patient dependence, time availability, and support. Next
experiments should add independent DGP families (including censoring,
nonstationary calibration, MNAR notes, and family-level shocks), tune only on
development seeds, compare kernel/local and calibrated baselines, report
family-level uncertainty, and separately validate Pythia coverage with its
actual boundedness/independence assumptions. Mayo remains downstream of signed
feature lineage and an auditable temporal OOF reproduction.

## Risk-card endpoint alignment attack

Code review found that caller-supplied Horos point-risk overrides could inherit
the cell-map's `mae_upper` and `probability_upper`. Those bounds concern the
cell-map estimand, not the substituted learned per-case forecast. The card
builder now emits `null` for each endpoint whose point quantity was overridden;
unmodified cell-map scores retain their endpoints. Regression checks cover
both paths. This removes a misleading interval, but does not add a calibrated
uncertainty interval for Horos forecasts—that remains future work. Focused
Metis/OOD checks: **14 passed**.

The novelty audit also includes [Riley et al., BMJ, *Uncertainty of risk
estimates from clinical prediction models*](https://www.bmj.com/content/388/bmj-2024-080749),
which reviews uncertainty around individual clinical risk estimates and
validation calibration. Therefore the project should distinguish its case
error-risk point forecast from uncertainty in the forecast itself and from a
Pythia certificate; those are separate outputs with separate evidence.
