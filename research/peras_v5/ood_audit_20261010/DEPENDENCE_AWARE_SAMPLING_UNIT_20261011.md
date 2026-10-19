# Dependence-aware sampling units for attainable-risk certificates

## Result

Pythia now changes its concentration unit according to the registered dependence graph of the repeated-environment data. Under arbitrary dependence among target coordinates, one environment block remains one independent score. Under a registered product design, every disjoint block--unit score is independent, so the certificate can use the full coordinate-level sample size.

This is not a heuristic effective-sample-size adjustment. It is a separate conditional theorem with an explicit coordinate-independence provenance requirement. The route refuses when that provenance is absent. A selection-safe router can compare row-level and coordinate-level families after dividing the declared error budget between them.

On fresh ten-mechanism batteries, the coordinate theorem raised median lower-bound tightness to 92--97% for MAE and pinball losses and substantially improved W15/W30/W60. It did not uniformly improve MSE because coordinate-wise score clipping can discard more high-variance mass than clipping after the unit average. The adaptive router correctly selected row-level MSE on the heavier-variance mechanisms.

## Theorem

Let environment rows be independent. Under the stronger product contract, all coordinates are also independent within each row, while coordinate `i` retains its registered conditional law `Q_i`. Partition environment rows into disjoint blocks `b` of size `k` and define

```text
V_bi = inf_a k^-1 sum_e loss(Y_ei,a).
```

For the supported empirical-optimum losses,

```text
E[V_bi] <= inf_a E_Qi loss(Y,a) = R_i*.
```

For MSE's unbiased-variance witness, equality holds before score truncation. Because `(b,i)` and `(b',i')` use disjoint independent coordinates whenever they differ, the full collection `{V_bi}` is independent, though generally not identically distributed. For a registered cap `c`,

```text
E[ average_bi min(V_bi,c) ] <= n^-1 sum_i R_i*.
```

Therefore a lower confidence endpoint for the mean of the flattened bounded scores is a lower endpoint for the fixed-design average Bayes risk. Hoeffding and empirical-Bernstein concentration apply directly; the repository's Bentkus inversion explicitly supports independent non-identically distributed bounded variables.

Under arbitrary within-environment dependence, flattening is invalid. In that case Pythia first averages units within each block and concentrates only across independent blocks. This is why the dependence contract changes the sampling unit rather than appearing as an informal variance correction.

## Fresh location-loss development audit

`run_coordinate_independence_ood.py` froze seeds 107001--107005, 800 target units, 5,000 product-coupled environments, and five losses. It compared the existing row-block theorem with the coordinate theorem using the same block sizes, caps, and loss-level error budgets.

| Loss | Row median bound/floor | Coordinate median bound/floor | Coordinate strict wins | Mechanisms won | Gold exceedances |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAE | 0.8591 | **0.9726** | 50/50 | 10/10 | 0 |
| MSE | **0.9657** | 0.9289 | 20/50 | 4/10 | 0 |
| Q10 | 0.7984 | **0.9435** | 50/50 | 10/10 | 0 |
| Q50 | 0.8591 | **0.9726** | 50/50 | 10/10 | 0 |
| Q90 | 0.7943 | **0.9201** | 50/50 | 10/10 | 0 |

Every coordinate location-loss route selected `k=32` and empirical Bernstein. MSE selected the unbiased-variance witness, but the per-unit cap truncated high-variance units before averaging; the row route applies the same numerical cap after averaging units and was therefore stronger on six mechanisms.

## Untouched adaptive-router audit

After observing the development exception, `environment_dependence_aware_portfolio` was frozen to split each loss's family error equally between the row and coordinate families and choose their maximum. `run_dependence_aware_ood.py` then used untouched seeds 108001--108005.

| Loss | Row-only median bound/floor | Adaptive median bound/floor | Adaptive strict wins | Selected routes | Gold exceedances |
| --- | ---: | ---: | ---: | --- | ---: |
| MAE | 0.8590 | **0.9723** | 50/50 | coordinate 50 | 0 |
| MSE | 0.9653 | **0.9658** | 20/50 | coordinate 20, row 30 | 0 |
| Q10 | 0.7977 | **0.9428** | 50/50 | coordinate 50 | 0 |
| Q50 | 0.8590 | **0.9723** | 50/50 | coordinate 50 | 0 |
| Q90 | 0.7942 | **0.9207** | 50/50 | coordinate 50 | 0 |

For MSE, the adaptive result is nearly identical in median to row-only and is not a run-wise dominance result: splitting alpha costs a small amount on the 30 runs that select row level. A preregistered loss-to-route map can avoid that selection cost, but it must be frozen on development data and evaluated on another untouched registry.

The ten mechanisms are hand-designed stress tests rather than an iid sample from a population of possible mechanisms. Winning all 10/10 is strong replication across the declared registry; `2^-10 = 0.0009766` is a descriptive sign reference, not proof of universal dominance.

## Fresh window-loss audit

`run_coordinate_windows_ood.py` froze seeds 109001--109005 before execution and evaluated window-miss Bayes risk.

| Loss | Row median bound/floor | Coordinate median bound/floor | Coordinate strict wins | Mechanisms won | Gold exceedances |
| --- | ---: | ---: | ---: | ---: | ---: |
| W15 miss | 0.6429 | **0.8404** | 50/50 | 10/10 | 0 |
| W30 miss | 0.3561 | **0.7731** | 50/50 | 10/10 | 0 |
| W60 miss | 0.0186 | **0.4972** | 50/50 | 10/10 | 0 |

All 150 coordinate bounds were positive. All selected `k=32`; empirical Bernstein won 145/150 concentration choices and Bentkus won five W60 cases. W60's gold is very small in several mechanisms, so the relative ratio must be read together with absolute risk.

## Innovation assessment

Using the correct independent sampling unit is classical probability, not a new concentration theorem. The scientifically interesting Pythia contribution is the **dependence-typed certificate interface**:

1. the estimand is unchanged across dependence routes;
2. a declared dependence graph determines which scores may be flattened;
3. each route exposes its provenance and assumptions;
4. adaptive comparison is compiled into a finite alpha ledger;
5. unavailable independence causes conservative routing or refusal rather than an optimistic effective sample size.

This is a credible component of a broader certificate-compiler paper, especially because the loss and cap can reverse which valid sampling unit is strongest. It is not evidence that routine clinical cases are mutually independent conditional-law replicates.

## Literature and next concentration gap

The current router uses Hoeffding, Maurer--Pontil empirical Bernstein, and Bentkus. Modern bounded-mean betting intervals can outperform older empirical-Bernstein bounds, and one-sided empirical intervals for U-statistics offer a principled path to reuse overlapping blocks. Relevant primary sources include [Waudby-Smith and Ramdas, *Estimating Means of Bounded Random Variables by Betting*](https://arxiv.org/abs/2010.09686), [Nguyen, *Concentration-based Confidence Intervals for U-statistics*](https://arxiv.org/abs/1903.01679), and [Pitcan, *A Note on Concentration Inequalities for U-statistics*](https://arxiv.org/abs/1712.06160).

These should be implemented as named external concentration baselines before claiming concentration optimality. They may reduce the remaining 3--20% gap without altering the risk witness, but their guarantees must cover independent non-identical scores or explicitly restrict the route.

## Mayo implication

Current Mayo data contain one realized outcome per surgery and do not provide independent repeated outcomes for each fixed patient-information state. Neither coordinate flattening nor the repeated-environment theorem can be applied to those rows. The result guides prospective simulator/replay or repeated-measurement design; it does not change the current 28.85 calibrated Mayo bound.

## Files

- Core: `repeated_environment_saa.py`
- Analytic checks: `check_repeated_environment_saa.py`
- Development runner/results: `run_coordinate_independence_ood.py`, `coordinate_independence_ood_results.jsonl`
- Untouched router runner/results: `run_dependence_aware_ood.py`, `dependence_aware_ood_results.jsonl`
- Window runner/results: `run_coordinate_windows_ood.py`, `coordinate_windows_ood_results.jsonl`

All rows are synthetic.
