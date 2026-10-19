# Direct average-W1 calibration: theorem candidate and fresh OOD audit

Resource-matched follow-up: `CALIBRATION_RESOURCE_FAIRNESS_20261010.md` implements the stronger calibration-data grouped SAA baseline flagged below. It beats this transport route in all 50 fresh runs for all three losses. Preserve the DKW improvement, but do not interpret it as superiority in the exact-group setting.

## Why change the calibration target?

Pairwise DKW caps pay uncertainty for every small group before averaging. The disjoint Pythia theorem needs only **one weighted mean** of group-law W1 distances, not a simultaneous table of individual caps. This note constructs a direct upper confidence cap for that mean using standard joint convexity and bounded differences. It is not a new concentration inequality and is not yet a literature-priority claim.

## Lemma and accounting

Let independent calibration samples of sizes `n_g` have common laws `P_g` within declared groups. Freeze clipping to a bounded interval of width `D`, group definitions, audit blocks, and all pair weights **without calibration or audit outcomes**. A design based on independent audit/calibration covariates is allowed, conditional on that design. Let `w_gh >= 0` be the total ordered within-block weight of the unordered distinct-group pair `{g,h}`; same-group terms are zero under the common-law assumption. Define

```
eta = sum_{g<h} w_gh W1(P_g,P_h)
A   = sum_{g<h} w_gh W1(P_hat_g,P_hat_h)
d_g = sum_{h != g} w_gh
r   = D sqrt( log(1/delta)/2 * sum_g d_g^2/n_g ).
```

Here all laws are clipped aligned laws, and each unordered weight already includes both ordered orientations. Joint convexity of W1 and unbiasedness of each empirical measure imply `eta <= E[A]`. Replacing one calibration observation in group g changes its empirical measure by W1 at most `D/n_g`, and therefore changes `A` by at most `D*d_g/n_g`. The **lower-tail** McDiarmid inequality gives `Pr(E[A] > A+r) <= delta`. Consequently

```
Pr( eta <= min(D*sum w_gh, A+r) ) >= 1-delta.
```

This cap is implemented by `calibrate_mean_block_w1_direct` in `independent_w1_calibration.py`. Its radius has no groupwise union cost. For balanced weights/counts, averaging can replace a penalty scaling with individual group size by one scaling with total calibration size. However, finite-sample empirical W1 remains upward biased and can itself become large; this is not a dimension-free accuracy claim. The proof fails if pair weights/groups/clips are optimized on the same calibration outcomes without new selection accounting. An adaptive Metis loop needs an independent selection split or simultaneous correction.

Use `delta=0.025` for the cap and `alpha_eval=0.025/3` for each of MAE, Q90, and MSE audit-score bounds. A union bound gives a joint at-least-95% conditional statement for these three losses in one dataset. The DKW comparator has its own same-size budget; **do not take an uncorrected maximum of the two routes**.

## Known-law regression check

On the existing two-group Bernoulli check (200 seeds; 20,000 independent calibration outcomes per group), direct cap median was 0.2446 versus DKW 0.3340; true ordered mean W1 was 0.1. Both had 0/200 cap misses. Q90 positive bounds increased from 159/200 to 198/200, with median bound 0.1257 versus 0.0451 against gold 0.99. Neither route exceeded known gold for any of the three losses. `check_independent_w1_calibration.py` now tests both routes.

## Untouched ten-mechanism audit

Before opening seeds 90001–90005, `run_direct_w1_ood.py` froze the following: the ten noise-law designs in `run_exact_law_battery.py`; 20,000 total calibration cases and 800 independent audit cases; observable exact-law key `(code,note concept)`; outcome-blind pairs formed by sorting concept, then code; clip `[0,300]`; score clips MAE=30, Q90=6, MSE=1000; the alpha allocation above. The same outcome-blind pair design and audit scores enter both cap methods. No DGP oracle W1 or predictive model enters the bound. The **equality of group conditional laws still comes from the synthetic design**, not a learned Mayo property.

| Loss | Direct positive / 50 | DKW positive / 50 | Direct strictly tighter / 50 | Direct median bound/gold | DKW median bound/gold | Direct gold exceedances |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE | 50 | 35 | 50 | 0.483 | 0.168 | 0 |
| Q90 | 32 | 4 | 32 | 0.038 | 0.000 | 0 |
| MSE | 5 | 0 | 5 | 0.000 | 0.000 | 0 |

MAE design-median tightness, direct versus DKW:

| Design | Direct | DKW |
| ---: | ---: | ---: |
| 1 | 0.572 | 0.535 |
| 2 | 0.502 | 0.183 |
| 3 | 0.411 | 0.000 |
| 4 | 0.515 | 0.305 |
| 5 | 0.487 | 0.153 |
| 6 | 0.440 | 0.000 |
| 7 | 0.491 | 0.106 |
| 8 | 0.476 | 0.247 |
| 9 | 0.528 | 0.398 |
| 10 | 0.281 | 0.000 |

`check_direct_w1_population.py` subsequently integrated the true clipped conditional CDF differences on a 12,001-point grid, solely as a diagnostic. The learned cap exceeded numerical population W1 in all 50 runs; minimum margin 0.1175 and median margin 0.7240. Numerical quadrature and observed zero misses are checks, not substitutes for the probability argument. Raw rows: `direct_w1_ood_results.jsonl`.

## Literature and remaining fairness gates

Wasserstein inference and concentration are established subjects, e.g. [Sommerfeld and Munk, finite-space inference](https://academic.oup.com/jrsssb/article/80/1/219/7048419) and [empirical-risk concentration via Wasserstein distance](https://jmlr.org/papers/volume23/20-965/20-965.pdf). [Jang and Noh (AISTATS 2026)](https://proceedings.mlr.press/v300/jang26b.html) study bias when minimizing empirical Wasserstein objectives; that work is a warning against treating optimized empirical distances as true law distances. Our fixed-weight upper-cap argument exploits upward bias rather than minimizing it. A full overlap/prior-art audit is still required before calling the average-cap application novel.

The 50/50 MAE gain is against **our conservative DKW cap route**, not against accepted Bayes-floor methods or SOTA. Under these same group-homogeneity and stationarity assumptions, the 20,000 calibration outcomes may also be used to estimate group risk floors directly; a fair calibration-data SAA/exact-replicate baseline could outperform the transport route and must be evaluated next. Report the calibration sample resource in every comparison. Likewise, same-group audit pairs need no W1 cap at all. Q90 and MSE remain weak, and none of this proves Mayo group homogeneity, absence of temporal drift, representation sufficiency, or patient-level independence. No Mayo result is changed or relabeled.
