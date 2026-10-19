# Resource-matched calibration baseline changes the interpretation

## Outcome

The direct-average W1 cap beat our pairwise-DKW route, but that comparison omitted a stronger use of the **same** 20,000 calibration outcomes. Under the same declared group-law equality and stationarity assumptions, estimating each group's risk floor directly is valid. On fresh seeds 91001–91005 across ten noise mechanisms, an elementary grouped eight-case SAA bound beat the transport bound in **all 50 runs for every supported loss**. This defeats a superiority claim for the transport route in the exact-group calibration setting.

## Established principle, explicit implementation

The principle `E[min_a empirical risk] <= min_a E[risk]` is standard sample-average approximation (SAA), not Pythia novelty. See [Mak, Morton and Wood (1999)](https://calhoun.nps.edu/bitstream/10945/38421/1/MakMortonWood.pdf), [Shapiro's statistical-inference treatment](https://epubs.siam.org/doi/10.1137/1.9780898718751.ch5), and [non-asymptotic optimal-value confidence bounds](https://arxiv.org/abs/1601.07592). `group_calibration_floor.py` implements this principle with explicitly proved Hoeffding/McDiarmid corrections; it is **not** the cited authors' reference implementation or a reproduction of every part of their protocol.

Let audit group weights `pi_g` be frozen without calibration outcomes and sum to one. Each group has one full conditional outcome law, unchanged from calibration to audit. With a frozen outcome clip of width D, define `V_g` as the full-group empirical optimal risk. Its expectation is no greater than the group's clipped Bayes risk. One calibration observation changes `sum pi_g V_g` by at most `pi_g A/n_g`, with `A=D` for MAE, `0.9D` for Q90, and `D²` for MSE. Therefore

```
full_SAA_bound = max(0, sum_g pi_g V_g
                         - A sqrt(log(1/alpha)/2 * sum_g pi_g²/n_g)).
```

Alternatively partition each group's calibration sample, in original sample order, into disjoint k=8 blocks. Clip each block's optimal risk at C. Let their mean be `S_g`, with `m_g=floor(n_g/k)`. Independent bounded scores and SAA optimism give

```
tuple_SAA_bound = max(0, sum_g pi_g S_g
                          - C sqrt(log(1/alpha)/2 * sum_g pi_g²/m_g)).
```

Discarding remainder **calibration** observations does not discard target audit weight: the retained blocks estimate the same group law. Outcome clipping contracts these location-loss Bayes risks, so both bounds lower-bound original risk. A missing or undersized positive-weight calibration group is refused. There is no predictive model fitting or storage; minimizing a finite empirical loss is the risk statistic itself. An original-outcome group empirical optimum is reported separately as a **biased point estimate without a confidence claim**.

## Frozen fair experiment

`run_calibration_resource_audit.py` uses exactly the transport study's ten DGPs, `(code,note concept)` law key, 20,000 calibration outcomes, 800 independent audit outcomes, `[0,300]` clip, and score caps MAE=30, Q90=6, MSE=1000. All group baselines use the **same calibration draws** as the W1 cap; audit covariates set group weights. The group baseline ignores audit outcomes, while transport uses them. Both have the same available information/resource budget. Each baseline uses `alpha=0.05/3` per loss, giving joint 95% coverage for that route's three losses; transport splits its 5% between calibration and audit. These are separate valid methods, not an uncorrected selectable maximum.

| Loss | Transport median bound/gold | Group tuple SAA | Full group SAA | Tuple wins / 50 | Tuple gold exceedances | Group plug-in median relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE | 0.486 | 0.839 | 0.812 | 50 | 0 | 0.0076 |
| Q90 | 0.035 | 0.678 | 0.322 | 50 | 0 | 0.0102 |
| MSE | 0.000 | 0.618 | 0.000 | 50 | 0 | 0.0186 |

Tuple SAA was positive in all 50 runs for all losses. Full SAA was positive in all MAE/Q90 runs, but only 5/50 MSE runs because its raw squared-loss bounded-difference radius is large. Full SAA exceeded gold 0 times as well. The plug-in errors benefit from 20,000 outcomes and a known sufficient law-group key: **do not compare them to the earlier 800-case, continuous-feature v5.7 estimator as if resources and assumptions were matched**.

`check_group_calibration_floor.py` adds 200 known-floor Bernoulli checks (zero exceedances for both bounds and all losses), translation invariance, and malformed-weight rejection. The theorem, not a zero-miss simulation, supplies the conditional guarantee. Raw OOD rows: `calibration_resource_audit_results.jsonl`.

## Revised research direction

1. Keep the direct-average W1 lemma as a useful tool, not a central SOTA result. Its 50/50 win against DKW did not survive the stronger resource-matched baseline.
2. Exact, sufficient groups should route to a direct grouped-floor method. Near-replicate continuous-feature risk needs a defensible heterogeneity correction; that remains the important unresolved problem.
3. Merely having a common **mixture** `Y|group` does not imply the group is sufficient for all raw features. Coarse-group Bayes risk can be *higher* than the raw-feature floor, so these results must not be certified for arbitrary feature engineering or future note readers without sufficiency/heterogeneity assumptions.
4. Reuse the existing Blackwell/information-set theory rather than relabeling the concavity of Bayes risk as a new definition of information. Any new contribution must quantify the within-group information gap with estimable, independently justified uncertainty.
5. Mayo grouping sufficiency, temporal stationarity, patient independence, and cap provenance remain unestablished. Neither this benchmark nor the prior cap result changes the Mayo claim tier or numeric anchors.
