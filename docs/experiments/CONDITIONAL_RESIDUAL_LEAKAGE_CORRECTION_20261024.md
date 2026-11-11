# Conditional residual-reader POC: leakage correction (2026-10-24)

**This note supersedes the validation claims in `docs/CONDITIONAL_RESIDUAL_EMBEDDINGS_20261024.md` until the experiment is rerun.**

## What was found

The testbed combines public MTSamples notes with a duration outcome planted from predeclared note concepts. It is useful as a known-truth simulation, not as observed clinical-duration evidence or Mayo validation.

The original FIT/CAL/TEST split was not respected by group residualization. In `cre_eval2.py`, the ridge target was centered by specialty means computed over the full array before the FIT rows were selected for training. The same full-array centering was used to fit stack weights. `cre_eval.py`'s duration-residual TEST R² routine used full-array target centering as well. Consequently, held-out outcomes could alter FIT-trained reader scores, calibration witnesses, and reported TEST metrics. Since Pythia's projection certificate requires a score fixed independently of CAL outcomes, the saved positive bound cannot be treated as valid evidence.

The repository `eb_scalar_bound` itself cannot detect target-adaptive scores. In a known-null adversarial simulation (zero within-cell conditional-mean spread), passing the leave-one-out centered CAL outcome as the score produced a positive lower bound in 1,000/1,000 replicates (median 0.15113). This violates the theorem's label-free-score premise and is a falsification of pipeline enforcement, not a counterexample to the theorem.

The reported 0.07190 vs 0.08490 is MAE for forecasting a synthetic W30-miss probability against its known conditional probability. It is **not** duration MAE in minutes and does not show that clinical notes improve the duration predictor's MAE. The stored 13.68097 min² lower bound and the “15% improvement” must not be presented as validated results.

## Repair applied

`research/instrumented_20261019/split_safety.py` adds a group-centering transform fit on training rows only. The CRE ridge fit and OOF submodels now use their own training-fold group means; the final reader is fit using FIT means; stack weights use FIT outcomes only. The duration-residual R² routine now uses FIT-only target/group transforms, with fold-specific transforms for its OOF models.

Regression tests verify that perturbing held-out outcomes cannot change training residualization or FIT-trained CAL scores. The fresh full test suite after the repair passed **489 tests, 9 skipped**. A 500-replicate known-truth Gaussian smoke test for the repaired Pythia path had 500/500 lower bounds below the CAL design's true within-cell conditional-mean variance; the median lower-bound/truth ratio was 0.037. This is one synthetic family, not a full coverage study.

## What remains before reporting results

The original reader cache and outputs are stale and must be regenerated. The raw MTSamples CSV is not available at the source path configured by the experiment in this checkout, so the reported reader and embedding results could not be recomputed after the fix.

Before a new result is called validated:

1. Restore the public source CSV and record its hash, row selection, and note-cache hashes.
2. Regenerate all note-derived readers and embeddings from that exact source.
3. Freeze all representation, tuning, capping, stack, and multiplicity choices using FIT only.
4. Fit residual scale for W30 probability forecasts from cross-fitted FIT errors, not in-sample residuals.
5. Run the unchanged CAL certificate exactly once, and evaluate duration MAE, W30 risk-probability MAE/Brier, and certificate coverage separately on untouched TEST.
6. Add target-in-text, shuffled-text, null-text, and wrong-stratum controls; preserve the known-truth DGP's analytic target and population risk.
7. Only then transfer to Mayo, using a strictly pre-cutoff note corpus and a patient-grouped temporal FIT/CAL/TEST design.

Separately, a synthetic attack showed that caller-selected `stratum` values could change the same case's estimate from 1.0/low to 50.0/high. `FrozenRiskMap.score` now refuses any non-aggregate caller-supplied stratum, and both Mayo Metis runners use the aggregate/prediction-supported route. Regression test `tests/test_metis_stratum_safety.py` covers the attack. This sacrifices subgroup-specific cards until a trusted case-input-to-stratum resolver is implemented.

The scorer still accepts prediction numbers that are not bound to an authenticated inference receipt. Thus signed model files alone do not prove which model produced the floats passed to `score`; all per-case forecasts remain conditional diagnostics, not verified per-case certificates. Mayo remains not leakage-cleared.

## Research positioning

Theoretical novelty is not established. Incremental information conditional on existing features has a long history in conditional-mutual-information feature selection; generic Bayes-risk lower bounds and conditional-risk calibration are also established areas. Pythia's potentially distinct target is a loss-specific finite-sample lower confidence bound on incremental attainable risk reduction under explicit support, dependence, and availability assumptions. This requires a systematic related-work review and a proof audit; the manuscript should not call explained variance itself “information” without defining the estimand and establishing its relationship to risk.
