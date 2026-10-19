# Independent group-law W1 calibration for the disjoint Pythia bound

Follow-up: `DIRECT_AVERAGE_W1_THEOREM_20261010.md` replaces the *pairwise DKW uncertainty accounting* with a direct cap on the required average. The DKW scale limitation below remains valid for the original method; it is not an impossibility theorem for all calibration routes.

## The gap

The disjoint-tuple theorem needs a valid upper cap on average within-block conditional-law Wasserstein-1 distance. Earlier OOD studies obtained that cap from the synthetic data-generating process, which is unavailable for Mayo. A string describing a cap's source is not a proof of the cap. This experiment supplies one narrow, fully auditable route from independent calibration outcomes, without fitting or retaining a predictor.

## Claim, assumptions, and proof

Assume groups `g=1,...,G` are declared before audit outcomes are seen and **every entity in group g has the same conditional residual-outcome law** `P_g`. Assume an independent iid calibration sample of size `n_g >= 2` from each law. Freeze a common monotone clip `C` to `[lo,hi]`, with width `D=hi-lo`. Let `Fhat_g` be the clipped empirical CDF. By the [DKW–Massart inequality](https://doi.org/10.1214/aop/1176990746) and a union bound, with probability at least `1-delta`, simultaneously for all groups,

```
sup_x |Fhat_g(x)-F_g(x)| <= eps_g
eps_g = sqrt(log(2G/delta)/(2n_g)).
```

In one dimension, `W1(P,Q) = integral |F_P-F_Q| dx`. Thus, on that **same simultaneous event**, every pair of different groups satisfies

```
W1(C#P_g,C#P_h) <= W1(Fhat_g,Fhat_h) + D*(eps_g+eps_h).
```

Clip the right side at `D`; same-group W1 is exactly zero **only under the declared group-law equality assumption**. Averaging the pairwise caps over all ordered within-block members supplies the exact `mean_w1_cap` required by `disjoint_transport_bound.lower_bound`. If the audit-score Bentkus event has failure probability `alpha_eval`, the full calibration-plus-audit claim has failure probability at most `delta + alpha_eval`, conditional on the stated group, independence, and fixed-choice assumptions. The implementation uses the more conservative simultaneous `delta=0.025` and `alpha_eval=0.025/3` for each of three losses; this yields a joint at-least-95% statement for all three loss bounds on one dataset. The conditional-law equality assumption is not itself statistically certified by DKW.

For MAE, Q90 pinball, and clipped MSE, outcome clipping is legitimate for a lower bound on **original** location-loss Bayes risk: for every action `a`, monotone contraction gives `ell(C(y),C(a)) <= ell(y,a)`, hence the clipped Bayes risk cannot exceed original Bayes risk. `disjoint_transport_bound.py` now supports a frozen outcome clip for MAE and Q90 as well as MSE. The calibration cap must describe the **clipped aligned laws** when clipping is supplied. Hard W30/W60 miss losses are still refused under this W1 route.

## Known-floor check

`check_independent_w1_calibration.py` freezes two observable groups whose outcomes are `{-10,+10}` Bernoulli with probabilities 0.50 and 0.51 for `+10`. Each run uses 20,000 **independent** calibration outcomes per group, then 400 fresh audit outcomes in 200 disjoint mixed-group pairs. This is intentionally a high-information calibration setting. Exact ordered within-pair mean W1 is 0.1. Across 200 seeds:

| Quantity | Result |
| --- | ---: |
| Median calibrated mean-W1 cap | 0.3340 |
| Cap misses of true mean W1 | 0/200 |
| MAE: known floor / median bound / positive / gold exceedances | 9.90 / 3.716 / 200/200 / 0/200 |
| Q90: known floor / median bound / positive / gold exceedances | 0.99 / 0.045 / 159/200 / 0/200 |
| MSE: known floor / median bound / positive / gold exceedances | 99.98 / 27.126 / 200/200 / 0/200 |

The 200-run check detects implementation failures; the probability guarantee comes from the explicit DKW, union, and audit-score argument, not from zero observed violations. Unknown groups, overlapping blocks, malformed clips, and missing or nonfinite calibration data fail validation. Reproduce with `python3 check_independent_w1_calibration.py` from this directory; the old `check_disjoint_transport_theorem.py` still passes unchanged.

## Ten-design calibration-scale stress test

The two-group example had 20,000 calibration outcomes **per** group. `check_group_calibration_scaling.py` instead spreads 20,000 total calibration cases across the actual `(procedure code, note concept)` exact-law groups in the existing ten synthetic OOD noise designs. It uses a predeclared `[0,300]` duration clip. For every adjacent-code, same-concept pair, it computes the **minimum** DKW-based mean-W1 penalty for a mixed pair by pretending empirical W1 is zero. Actual empirical W1 can only increase this penalty. Thus this is a favorable diagnostic for the calibrated route, not a measured bound result.

| Design | Groups | Median minimum DKW penalty / known original MAE floor | Optimistic total calibration `n` to make DKW penalty smaller than floor |
| ---: | ---: | ---: | ---: |
| 1 | 30 | 1.41 | 41,280 |
| 2 | 120 | 2.44 | 123,840 |
| 3 | 240 | 3.40 | 240,960 |
| 4 | 60 | 3.03 | 195,840 |
| 5 | 120 | 2.22 | 101,880 |
| 6 | 180 | 3.21 | 214,920 |
| 7 | 120 | 2.70 | 149,040 |
| 8 | 120 | 1.51 | 47,160 |
| 9 | 60 | 1.91 | 75,540 |
| 10 | 240 | 4.72 | 457,200 |

The final column assumes **balanced** groups and solves `D*sqrt(log(2G/delta)/(2n_g)) < known original MAE floor` with `D=300`; it ignores actual between-group W1, imbalanced group sizes, score-concentration loss, and the fact that clipping may reduce the floor. It is therefore an optimistic threshold, not a recommended sample size. The ratio above one does not mathematically force every realized bound to zero, but it shows why the present DKW route is not a plausible broadly useful Mayo certificate at available calibration sizes. Reproduce with `python3 -c 'from check_group_calibration_scaling import run; run(clip=(0.,300.))'` from this directory.

## What remains unsolved

This is **not a claim of a new Wasserstein confidence inequality**; [finite-sample one-dimensional transport inference](https://tmanole.github.io/files/thesis.pdf) and DKW are established. The Pythia-specific contribution under investigation is their use as an independently accounted cap inside a Bayes-floor tuple witness. Even that application needs comparison with tighter two-sample bands and direct risk-floor methods before a novelty claim.

The cap scales roughly with `D/sqrt(n_g)` per group, so many small procedure/note groups or a wide duration clip make it vacuous. Its strongest assumption—that a declared group is outcome-law homogeneous and stationary from calibration to audit—is not proved merely by matching codes or note concepts. Continuous covariates, temporal drift, and patient dependence require new simultaneous calibration or sensitivity theory. No Mayo certificate is released from this experiment, and no oracle-cap result is relabeled as deployable.
