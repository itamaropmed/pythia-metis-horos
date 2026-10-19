# Fathom on Mayo HRS: diagnosis and an improved bound

All quantities in **minutes**. n = 12,993 cases (baseline-v2 exclusions, 2023-01-01 burn-in), 10,532 patients.
Cross-fitting: patient-grouped 5-fold, two distinct partitions, six out-of-fold predictors
(HGB `absolute_error`, XGBoost `reg:absoluteerror`, ExtraTrees).

## 1. Where the looseness was

| step | arm B (tabular + notes) | cost |
|---|---|---|
| estimate of the floor, Fathom-E | 32.31 | — |
| bound before any guard | 31.09 | 1.22 (the sqrt2 safety factor on MAD(e)) |
| bound with the published 1-D guard | 26.22 | **4.87 (the guard)** |

The guard was **80%** of the total looseness. It fires at qhat = 1.1611 because the residual shape ratio is rho = 0.669, deep in the heavy-tailed part of the envelope.

## 2. The defect this exposed

Every one of the 496 pairs in the published shape library was simulated with **both components standardised**, i.e. at component scale ratio 1. Mayo is nowhere near that: the model-error share of the residual is

    tau = (M+/sqrt2) / MAD(r) = 0.229

so the noise component dominates the model-error component about 4:1. The quadrature excess collapses as the components become unequal — on the worst library pair q falls from 1.252 at tau=1/2 to 1.10 at tau=0.19 — so the published guard was calibrated at the single worst scale ratio and then applied at a benign one.

Two corrections follow.

**(a) A sharper universal guard cap, proved.** Subadditivity gives MAD(U) >= MAD(U+V) - MAD(V), hence

    q <= [ (1-tau)^2 + tau^2 ]^(-1/2) =: Qcap(tau)

which equals sqrt2 exactly at tau = 1/2 and is strictly smaller elsewhere. This guard inequality needs no simulation. The complete floor bound remains conditional on the assumption that disagreement captures the non-shared model-error component; see `docs/experiments/REAL_WORLD_VALIDATION.md`.

**(b) A two-dimensional envelope qhat(rho, tau),** rebuilt from the same 31-law library over a grid of 12 scale ratios (5,952 measured combinations), with the same rules as before: 95th percentile in a local window of >= 30 combinations, monotonised (non-increasing in rho, non-decreasing in tau), floored at 1 and capped at Qcap(tau).

## 3. Re-validation on the synthetic suites

| guard | suite K (20 adversarial) | | suite P (30 prospective) | | pooled |
|---|---|---|---|---|---|
| | valid | usable | valid | usable | valid |
| 1-D envelope (published) | 20/20 | 0.694 | 30/30 | 0.846 | 50/50 |
| **2-D envelope (rho, tau)** | 20/20 | **0.721** | 30/30 | **0.887** | 50/50 |
| Qcap(tau), proved guard / conditional full rung | 20/20 | 0.561 | 30/30 | 0.634 | 50/50 |
| sqrt2, proved guard / conditional full rung | 20/20 | 0.486 | 30/30 | 0.575 | 50/50 |

The 2-D guard is **not** a uniform loosening: on the five suite-K designs with tau > 0.45 (K01, K02, K06, K17, K19) it is *larger* than the 1-D guard and the bound gets smaller. Worst-case ratio on suite K rises from 0.918 to 0.966, so the safety margin is thinner while validity holds on all 50 designs.

## 4. Second improvement: stratified Fathom

R* = E[MAD(Y|X)] is additive over any partition S = s(X): R* = sum_s w_s R*_s. So if B_s <= R*_s in every stratum, sum_s w_s B_s <= R*. Within a stratum we use MAD_s(r) rather than the mean absolute residual, which makes the chain exact: (MAD_s(r)/qhat_s)^2 - MAD_s(e)^2 <= MAD_s(eps)^2 <= R*_s^2, the last step because MAD_s(eps) = min_t E[|eps-t| | s] <= E[|eps| | s] = R*_s. This is a theorem, not a re-calibration, so it needs no re-validation.

Partition pre-specified as quintiles of the out-of-fold prediction. Sensitivity over K, arm B: K=2: 28.72  K=3: 28.79  K=5: 28.85  K=10: 28.81 — flat, so nothing is being tuned.

## 5. Results

| arm | p | L (best OOF MAE) | M+ | rho | tau | **B\* published (1-D)** | **B\* improved (2-D + strat)** | Qcap rung | sqrt2 rung | **E (estimate)** | temporal holdout MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tabular only | 296 | 33.32 | 10.96 | 0.667 | 0.233 | 26.52 | **29.29** | 24.55 | 20.94 | 32.77 | 32.28 [30.89, 33.50] |
| tabular + 11 note features | 307 | 32.87 | 10.67 | 0.669 | 0.229 | 26.22 | **28.85** | 24.19 | 20.63 | 32.30 | 31.67 [30.36, 32.85] |
| tabular + notes (features + text) | 409 | 33.82 | 10.41 | 0.669 | 0.218 | 27.20 | **30.01** | 25.47 | 21.51 | 33.22 | 32.01 [30.69, 33.22] |

### Per-stratum detail, arm B (tabular + notes)

| quintile of predicted duration | n | MAD(r) | rho | tau | qhat(rho,tau) | M+ | B\* |
|---|---|---|---|---|---|---|---|
| 1 | 2599 | 14.6 | 0.672 | 0.260 | 1.0847 | 5.4 | 12.4 |
| 2 | 2598 | 24.2 | 0.680 | 0.225 | 1.0820 | 7.7 | 21.0 |
| 3 | 2599 | 30.0 | 0.691 | 0.245 | 1.0813 | 10.4 | 25.7 |
| 4 | 2598 | 39.4 | 0.731 | 0.222 | 1.0682 | 12.4 | 34.7 |
| 5 | 2599 | 55.8 | 0.757 | 0.216 | 1.0484 | 17.1 | 50.4 |

## 6. What it means for the notes arm

- Calibrated floor on the best attainable MAE: **28.8 min** (was 26.2).
- Conditional Qcap-guard rung: **24.2 min** (was 20.6 under sqrt2). The guard is proved; the full bound still has an explicit shared-error assumption.
- Estimate of the floor: **32.3 min**.
- Best out-of-fold MAE achieved: 32.9 min.

So at most **4.0 min** of the current 32.9-min out-of-fold error is provably recoverable, down from 6.6 min under the published guard; the point estimate says the true recoverable slack is about 0.6 min.
