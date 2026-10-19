# Variance-stabilized SIMEX: frozen OOD negative control (2026-10-10)

## Question and scope

Frozen Pythia v5.7 extrapolates a weighted average of in-tuple and leave-one-out loss statistics linearly in `1+lambda`. Could squaring the statistic before extrapolation reduce point-estimation bias for MAE and Q90? This is **only a point-estimator candidate**. It changes neither the existing lower-bound construction nor the transport theorem, and it is not a confidence procedure.

For a centered Gaussian location-scale tuple with **isotropic independent Gaussian** extra noise of variance proportional to `lambda`, a degree-one homogeneous loss statistic has expected value proportional to the square root of total variance. Its squared population curve can therefore be linear in that variance. This idealization is not the actual Pythia setting: `pythia5.simex_draw` adds a shared random slope propagated through member-specific feature offsets, producing anisotropic and correlated increments. Moreover, the v5.7 statistic mixes two risks before squaring, while heterogeneous neighborhoods and non-Gaussian residual laws destroy the simple square-root scaling. Thus the transformation is a testable approximation, not a theorem for Pythia.

## Frozen experiment

`simex_variance_stabilized.py` duplicates the v5.7 row calculation without changing its frozen source and computes both the original linear extrapolation and the square-transform variant on **identical random draws**. `run_variance_simex_audit.py` checks direct numerical parity against `pythia5.pythia57` before running. All printed estimator components matched exactly in the checked design/seed. Neither route fits a predictive model. The ten mechanisms and known Bayes risks are from the prespecified `audit_ood.py`; each has 800 cases.

Development used seeds 27001–27003 (already used in the prior OOD audit). The rule `square MAE and Q90; do not alter the other losses` was then evaluated, without adjustment, on untouched seeds 38001–38005. Design-level results are medians over five seeds. Relative error is `abs(estimate - known Bayes risk) / known Bayes risk`.

| Loss | Linear median relative error, 50 runs | Squared median relative error | Squared wins, 50 paired runs | Squared wins, 10 design medians |
| --- | ---: | ---: | ---: | ---: |
| MAE | 0.0554 | 0.0606 | 28 | 5 |
| Q90 | 0.0777 | 0.0785 | 28 | 4 |

For MAE, squared extrapolation improved the checkerboard, smooth heteroscedastic, sparse, skew, and nonlinear-heavy-tail design medians, but worsened linear Gaussian, Friedman Gaussian, text interaction, text mixture, and discrete-replicate designs. For Q90, it improved 4/10 design medians. The design-level one-sided sign-test probabilities are 0.623 for MAE and 0.828 for Q90; these are descriptive because mechanisms were deliberately specified rather than randomly sampled from a task population. The numerical improvement on some difficult mechanisms is not a general gain, and the across-run median is worse for both losses.

## Decision and next gap

**Do not promote this transform to Pythia.** Preserve frozen v5.7 as the point-estimator baseline and report the negative control. The experiment isolates a theoretical gap: a correct SIMEX continuation for local detrending has to account for the **full induced-trend covariance** and the nonlinearity of the loss-specific tuple functional, rather than substituting a scalar noise variance. A future candidate should derive or compute the expected statistic under that covariance without using outcome-oracle knowledge, predeclare its route, and face another untouched OOD suite with external point-estimation baselines. It must never be used to repair the one-sided certificate by analogy.

This gap is not a novelty claim about SIMEX itself. Existing measurement-error work explicitly handles [heteroscedastic contamination](https://arxiv.org/abs/0902.2117) and [correlated spatial error](https://pmc.ncbi.nlm.nih.gov/articles/PMC4834950/). Those settings reinforce the need to state and validate an error-covariance model; they do not establish that our tuple-risk continuation is identified or that it yields a Bayes-floor bound. Nor should the [general Bayesian estimation-risk lower bounds](https://www.jmlr.org/papers/v17/16-185.html) be treated as a matched benchmark for a fixed data-distribution prediction floor without checking estimand and protocol.

Raw paired rows: `variance_simex_development_results.jsonl` and `variance_simex_fresh_results.jsonl`. Reproduction: `python3 run_variance_simex_audit.py --first-seed 38001 --last-seed 38005 --output /private/tmp/pythia_simex_fresh.jsonl` from this directory.
