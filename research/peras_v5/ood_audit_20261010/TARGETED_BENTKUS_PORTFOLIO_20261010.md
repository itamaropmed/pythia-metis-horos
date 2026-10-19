# Targeted randomized Bentkus portfolio: a modest MSE upper improvement

## Question

The eight-loss grouped intervals use weighted Hoeffding concentration. Their MSE upper endpoint is especially wide because squared held-out loss ranges over `D²`. Could the sharper [Bentkus inequality](https://arxiv.org/abs/math/0410159) improve it while preserving the finite audit-group mixture and joint selection accounting?

Bentkus in the existing implementation applies to an **equal-weight mean** of independent bounded scores, which is not the ordinary unequal-weight grouped statistic. Two routes were investigated without changing the clipped group-risk estimand.

## Routes and validity scope

### Exact balanced allocation

For audit group counts `c_g`, retain `r*c_g` disjoint calibration blocks from every group using the largest common integer r. The resulting equal-weight block sequence has exactly the audit proportions. This route is deterministic conditional on the outcome-blind group counts and sample sizes. It refuses if any group lacks even `c_g` blocks. With k=16 and 20,000 total calibration cases, it issued on only **6/50** development datasets because rare, imbalanced groups exhausted their blocks. Reducing k to 8 improved capacity but worsened interval tightness. The route remains implemented as `balanced_bentkus_interval`, but is not a practical default.

### Registered random mixture

Draw a fixed number B of group labels iid from the audit group proportions using a seed registered before any calibration outcomes. Each scheduled label consumes a fresh, disjoint k-case block from that group. Marginally over this group randomization and calibration outcomes, block scores are independent draws from the audit mixture, so Bentkus can be inverted for the mixture expectation. If a scheduled group exceeds available capacity, refuse—**never redraw, search seeds, truncate the schedule, or adapt B after inspection**.

This is an unconditional repeated-procedure guarantee over the registered randomization and outcomes, conditional on the structural assumptions. It is not automatically 95% coverage *conditional on a certificate being issued*. Failures issue no numeric claim, hence the probability of issuing a false claim remains bounded by the original event probability. A deterministic hand-picked schedule cannot inherit the mixture argument. The seed and scheduled counts are certificate artifacts, not tuning knobs.

For a lower endpoint, invert Bentkus directly on clipped SAA scores. For an upper endpoint with held-out score `H in [0,R]`, invert a lower bound for `E[R-H]` and return `R-LB`. No upper witness is winsorized below its true range.

## Development finding

With k=16, B=200, and development seeds 94001–94005, the random route issued on 40/50 datasets. It improved the MSE upper endpoint against same-seed weighted Hoeffding in 39/40 issued cases; several pinball upper endpoints improved too. But it worsened MAE, median, and most window interval widths because it uses only 200 blocks. With k=8 it issued 50/50 but was wider across nearly every loss. Therefore only the **MSE upper endpoint** was predeclared for the final portfolio.

## Selection-safe fresh portfolio

`run_targeted_multiloss_portfolio.py` freezes:

- all eight losses and both weighted-Hoeffding endpoints: 16 events;
- one additional registered-randomized Bentkus MSE upper event;
- `alpha_event=0.05/17` for every event;
- k=16, B=200, the existing clip `[0,300]`, caps, ten exact-law synthetic mechanisms, and seeds 95001–95005;
- MSE upper = minimum of its jointly covered Hoeffding and Bentkus endpoints when the latter issues; otherwise Hoeffding.

The union bound makes this 17-event portfolio jointly at least 95% in its unconditional randomization/outcome sense, under the group-law, stationarity, independence, and frozen-choice assumptions. It must not be combined post hoc with the earlier alpha/16 route.

| Fresh result | Value |
| --- | ---: |
| Datasets | 50 |
| Randomized route issued | 40/50 |
| Randomized route selected when issued | 40/40 |
| Known clipped-gold endpoint misses, all 800 endpoints | 0 |
| Median baseline MSE upper/gold (`alpha=.05/16`) | 8.078 |
| Median portfolio MSE upper/gold | **6.994** |
| Median portfolio/baseline MSE interval width | **0.850** |
| Portfolio MSE upper smaller than baseline | 39/50 |
| Portfolio MSE upper larger than baseline | 11/50 |

The 11 larger cases reflect the alpha/17 cost and capacity refusals; the portfolio is not pointwise dominant over the separately budgeted alpha/16 baseline. Other loss widths increased by roughly 0.2–0.5% from the extra union event. Raw rows: `targeted_multiloss_portfolio_results.jsonl`; development rows: `randomized_bentkus_development_results.jsonl`. `check_finite_group_multiloss.py` now includes 200 two-point known-floor checks for the exact and randomized routes, with zero observed misses over 1,600 loss cases per route. Simulation checks do not replace the concentration proof.

## Interpretation

This is a useful engineering refinement, not a broad theoretical novelty. Bentkus already covers independent, potentially nonidentical bounded variables; proportional/randomized allocation is classical sampling practice. The Pythia-specific value is honest routing and alpha accounting across loss endpoints, with refusal when the schedule cannot be executed. Even after the improvement, the MSE upper endpoint remains very loose, and all results concern a clipped finite-group representation—not Mayo raw features, unbounded duration risk, or every future model.
