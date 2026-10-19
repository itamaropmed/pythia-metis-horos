# Heterogeneous-support Chernoff certificate

Status: **conditional theorem implementation; fresh synthetic validation passed**  
Scope: frozen Pythia NLI witnesses, independent labels conditional on the design  
Not a claim of a new concentration theorem or universal state of the art.

## Why this was needed

For a frozen score direction `h_i`, the NLI witness observes `Z_i = h_i[Y_i]`,
with outcome-blind support `[l_i, u_i]`. The earlier certificate used the exact
row ranges `r_i = u_i-l_i` in a Hoeffding radius. That was stronger than the
repository's Bentkus and empirical-Bernstein routes on all ten generated
table+text families because those routes replaced every `r_i` by the cohort
maximum and discarded support geometry.

## Construction

Shift `X_i = Z_i-l_i`, so `0 <= X_i <= r_i`. For a null total mean `M`, the
Chernoff bound is

`P(sum X_i >= s) <= inf_{lambda>0} exp(-lambda s) sup product_i E exp(lambda X_i)`.

The chord bound for the exponential says that, at a fixed mean, each factor is
maximized by an endpoint Bernoulli `X_i = r_i B_i`. The remaining inner problem
allocates probabilities `p_i` subject to `sum r_i p_i <= M`. Its log-objective
is concave and its KKT solution is

`p_i = clip(1/(eta r_i) - 1/(exp(lambda r_i)-1), 0, 1)`,

where `eta` enforces the mean constraint. Pythia numerically minimizes over
`lambda` and inverts the resulting tail envelope in `M`.

This retains every row's support width, allows independent non-identically
distributed labels, and uses no outcome-adaptive geometry. The theorem is
conditional on the score matrix being frozen without the certified labels.

### Pointwise dominance over heterogeneous Hoeffding

For every feasible mean allocation, Hoeffding's lemma gives

`log E exp(lambda (X_i-E X_i)) <= lambda^2 r_i^2 / 8`.

Therefore the optimized endpoint-Bernoulli mgf used above is no larger than
`lambda M + lambda^2 sum_i r_i^2/8`. Minimizing the latter yields the standard
heterogeneous Hoeffding tail. Hence the exact Chernoff tail envelope is
pointwise no looser, and its inverted lower endpoint weakly dominates the
heterogeneous Hoeffding endpoint for every observed sample. The implementation
uses the Hoeffding endpoint as a numerical floor; this follows the proved
ordering and is not post-outcome method selection.

Implementation:

- `pythia/core/concentration.py::heterogeneous_chernoff_lower_sum`
- `pythia/core/unified.py::nli_cohort_bound(..., concentration="heterogeneous_chernoff")`

## Fresh results

```bash
python3 research/peras_v5/unified_20261018/compare_nli_concentration.py 2500 106001 5
python3 research/peras_v5/unified_20261018/compare_nli_concentration.py 2500 706001 5 ood
```

| frozen suite | datasets | informative | strict wins vs row-Hoeffding | losses | median bound ratio | validity violations |
|---|---:|---:|---:|---:|---:|---:|
| ten table+text mechanisms, five fresh seeds | 50 | 40 | 40 | 0 | 1.0191 | 0 |
| six separate OOD language families, five fresh seeds | 30 | 25 | 25 | 0 | 1.0131 | 0 |

The remaining datasets tied at zero. The first suite's 40/40 direction has a
descriptive one-sided sign probability `2^-40 = 9.09e-13`; the OOD suite has
`2^-25 = 2.98e-8`. These are development comparisons, not preregistered
confirmatory p-values. The practical gain is incremental (median 1.3%--1.9%),
not transformative.

The selection-safe portfolio remained valid but was weaker because it pays for
five simultaneous routes. A support-stratified Bentkus attempt was also valid
but lost: splitting alpha across bins cost more than local caps saved. It is a
negative control, not the recommended route.

### Untouched 10,000-draw concentration audit

Ten additional designs used 12--39 independent endpoint-Bernoulli variables,
log-uniform caps from roughly `0.025` to `1.8`, and heterogeneous beta-drawn
success probabilities. Across 10,000 untouched draws:

- coverage: **0.9971**;
- 95% Wilson lower endpoint for coverage: **0.99584**;
- violations: **29/10,000** at nominal alpha `0.05`;
- strict wins over heterogeneous-range Hoeffding: **9,979/10,000**.

The frozen summary is
`docs/results/heterogeneous_chernoff_validation_20261018.json`.

## Falsification and regression checks

- scale equivariance and permutation invariance;
- zero observation returns zero;
- 160 endpoint-Bernoulli repetitions with heterogeneous caps;
- known-law checks on all 80 generated datasets above;
- all compared routes had zero observed target violations;
- malformed caps and unknown concentration routes fail closed.

## Literature position

This construction composes classical endpoint extremality and Chernoff
inversion with Pythia's frozen NLI witness. Relevant primary references are
[Hoeffding (1956)](https://doi.org/10.1214/aoms/1177728178),
[Bentkus (2004)](https://doi.org/10.1214/009117904000000360), and
[Maurer and Pontil (2009)](https://arxiv.org/abs/0907.3740). The defensible
novelty candidate is the support-aware residual-information certificate and
its model-free composition, not the general concentration ingredients.

## Remaining gates

1. Independent proof audit of the endpoint reduction, KKT allocation, and
   numerical inversion, including optimizer tolerances.
2. Exact small-`n` enumeration against the implemented envelope.
3. At least 10,000 untouched repetitions with Wilson coverage reporting.
4. Patient-cluster and repeated-measure extension before Mayo promotion.
5. Keep all score/model fitting in Horos. The migration is now implemented in
   `horos/models/pythia_adapters.py` and enforced by `test_architecture.py`.
6. Compare the final Bayes-floor estimand, not merely concentration routes,
   against external published attainable-error protocols.
