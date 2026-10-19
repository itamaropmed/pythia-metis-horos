# Non-oracle repeated-environment modulus calibration

## Result

Pythia now has a production route that calibrates the barycentric discrepancy
from genuinely independent repeated environments.  It does not estimate a
smoothness constant from ordinary observational rows.  Instead, every
environment provides one coupled draw from each registered unit-conditional
law; dependence within a row may be arbitrary, while rows must be iid.

For each environment and disjoint block, the observed costs are

```
C1 = mean_j |V_j - Vbar|
C2 = mean_j (V_j - Vbar)^2.
```

Their expectations are valid penalties for MAE/pinball and MSE respectively
under the barycentric coupling theorem.  On a clip of width `D`, `C1` lies in
`[0,D/2]` and `C2` in `[0,D^2/4]`.  Simultaneous upper-mean endpoints therefore
give non-oracle caps on the aggregate discrepancies.  Dividing by frozen
feature geometry gives effective aggregate moduli `P/G` and `sqrt(P2/H)`;
these are not uniform pairwise Lipschitz constants.

## Development audit

- ten registered affine-quantile mechanisms;
- 80 fixed units, 20 disjoint blocks of four;
- 500 common-uniform target environment rows;
- seeds `133001--133100`, 1,000 runs;
- Hoeffding, empirical Bernstein, and Bentkus under one alpha ledger.

| Penalty | Misses | Positive-truth runs | Median cap/truth | 90th percentile |
| --- | ---: | ---: | ---: | ---: |
| Absolute | 0/1,000 | 900 | 2.808 | 3.371 |
| Squared | 0/1,000 | 900 | 57.733 | 153.429 |

The squared endpoint exposed a major range problem.  The absence of misses did
not make it practically tight.

Raw SHA-256:
`89189caa9ed2ed06c5e36128780ef68cbfbfdc1ba158ec0c55b8690123d95348`.

## Frozen V1 router and untouched confirmation

Development evidence selected dKelly for absolute costs and Bentkus for
squared costs.  V1 froze these choices before opening seeds
`134001--134100`; alpha `.05` was split equally across the two endpoints.

| Penalty | Misses | Wilson 95% lower validity | Median cap/truth | 90th percentile | Development median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Absolute | **0/1,000** | 0.9962 | **1.996** | 2.450 | 2.808 |
| Squared | **0/1,000** | 0.9962 | **47.020** | 124.629 | 57.733 |

Raw SHA-256:
`1ceeafc59c68b96e7422fac628b49015cd3f38caab4f601f1cc500e1f37970ea`.

The confirmation supports validity and the absolute-loss improvement.  It
also confirms that MSE remains range-limited rather than solved.

## Support-aware range theorem and second untouched confirmation

For a frozen post-shift hard support `[l_i,u_i]` for every unit, both
barycentric costs are convex functions of a block outcome vector.  Their
maximum over the rectangular support is therefore attained at a vertex.
Because blocks are disjoint, the exact range of the average environment cost
is the average of the per-block vertex maxima.  `support_aware_barycentric_cost_ranges`
enumerates these vertices, refusing blocks larger than the registered
computational limit.

The route was frozen before seeds `135001--135100`.  The ten synthetic
mechanisms provide analytic unit supports; all other protocol details stayed
unchanged.

| Penalty | Misses | Wilson 95% lower validity | Median cap/truth | 90th percentile | Global-range median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Absolute | **0/1,000** | 0.9962 | **1.091** | 1.162 | 1.996 |
| Squared | **0/1,000** | 0.9962 | **2.009** | 3.697 | 47.020 |

Raw SHA-256:
`552b40431c421514803f3e66f976cdf260644a377f369c1ef6f8bbb85f98dca0`.

This closes most of the synthetic squared-range gap but does so with a strong,
auditable resource.  Analytic simulator supports are available in this audit;
ordinary fitted prediction intervals are not hard supports and cannot be
substituted.  Current Mayo only has a broad operational duration domain, so no
clinical number is tightened by this result.

## Executable certificate composition

`calibrated_barycentric_lower_bound` now combines a simultaneous block-score
LCB with the matching repeated-environment penalty UCB.  It:

- checks a content hash of the block design on both evidence objects;
- applies the exact MAE, one-half pinball, or squared-loss penalty;
- combines score and calibration failure probabilities by a union bound;
- records distinct event identifiers and provenance;
- refuses mismatched block designs or incomplete calibration.

Thus the new calibration is not only a diagnostic modulus estimate; it can
enter the certificate graph as a typed lower endpoint.  This does not weaken
the repeated-environment or hard-support assumptions.

## Resource fairness

When environment coordinates have the exact target unit-conditional laws,
the environments are also exact target replicates.  Direct repeated-
environment SAA can use that resource without a transport penalty and must be
reported as a comparator.  The calibration route is most scientifically
useful for:

1. validating a declared discrepancy;
2. composing an externally justified upper-envelope environment with an
   ordinary evaluation cohort;
3. diagnosing how much of a conditional certificate is carried by the
   discrepancy assumption.

An external simulator is not automatically an upper envelope.  That transfer
is a structural assumption and receives an explicit marginal-contract field.

## Relation to modern research

This route uses classical bounded-mean inference and coupling, not a newly
claimed concentration inequality.  Recent Wasserstein research develops
finite-sample inference and highlights substantial statistical complexity,
including finite-sample bias of empirical Wasserstein objectives.  The Pythia
contribution is the loss-specific, provenance-bearing composition into an
attainable-risk certificate, together with fail-closed resource contracts.

Relevant primary sources:

- [Shah and Peters, hardness of conditional-independence testing](https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-3/The-hardness-of-conditional-independence-testing-and-the-generalised-covariance/10.1214/19-AOS1857.full)
- [Wang, Boedihardjo and Xie, finite-sample guarantees for Wasserstein distances](https://proceedings.mlr.press/v267/wang25g.html)
- [Jang and Noh, finite-sample bias of empirical Wasserstein minimization](https://proceedings.mlr.press/v300/jang26b.html)

## Mayo consequence

Mayo has one observed outcome per case, not repeated target environments for
the same registered units.  This route must therefore refuse on the current
cohort.  A simulator/twin may be used as an external upper-envelope only after
its transfer contract is justified; a twin-implied cap remains calibrated,
not proved.  No retrospective Mayo headline changes.

## Next theorem work

1. Develop a valid soft-tail/moment analogue for settings without hard
   unit-level supports; the hard-support geometry route is now implemented.
2. Extend the repeated-environment penalty to cluster/time-dependent rows.
3. Develop the multi-point pinball witness under the same alpha ledger.
4. Test whether held-out environments can falsify, rather than prove, an
   external Mayo modulus declaration.
