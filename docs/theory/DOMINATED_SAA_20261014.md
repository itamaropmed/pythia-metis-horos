# Fixed-action domination: full-sample SAA floor certificates without blocks or support

## Problem it closes

`CLUSTER_CONCENTRATION_GAP_20261014.md` identified a gap: Pythia certified the nonlinear full-sample SAA optimum only by one of two routes, each with a cost.

| Route | Cost |
|---|---|
| McDiarmid with a hard outcome range | Needs support; the global range is loose |
| Disjoint block-SAA plus betting | Block optimism; only G/k independent scores |

The gap note correctly said that Bentkus, empirical Bernstein and betting cannot be applied to the SAA optimum *as if it were an additive mean*.

This note shows that range-based inequalities can nevertheless be applied, *one-sidedly*, to an unobserved additive sum that dominates the optimum.

## Theorem

**Setting**
- Independent entity clusters g = 1..G. Dependence between cases inside a cluster is arbitrary.
- Cells c(i) frozen without outcomes.
- A nonnegative loss ℓ and a cap κ > 0.
- Bayes actions a\*_c, which exist for MAE, pinball and window-miss and are unknown.
- Outcome-blind retention of at most m_g cases per cluster.

**Definitions**
- The observable statistic is S = (1/n) Σ_c min_a Σ_{i ∈ c, retained} min(ℓ(Y_i − a), κ).
- The unobservable cluster terms are W_g = (G/n) Σ_{i ∈ g, retained} min(ℓ(Y_i − a\*_{c(i)}), κ), with b_g = κ·G·m_g/n and R = n⁻¹ Σ_i R_{c(i)}.

**Claims**
1. **Pathwise domination.** S ≤ W̄ := G⁻¹ Σ_g W_g, because each cell minimum is at most the value at a\*_c.
2. **Independence and boundedness.** The W_g are independent, with W_g ∈ [0, b_g].
3. **Mean below the target.** E W̄ ≤ R. Truncation and dropping cases remove only nonnegative loss.
4. **Tail transfer.** For every t, P(S − R ≥ t) ≤ P(W̄ − E W̄ ≥ t).

**Consequences**
- **Hoeffding:** R ≥ S − sqrt(log(1/α) Σ b_g² / (2G²)).
- **Bentkus**, with common range B = max_g b_g: R ≥ B·p_L, where p_L solves e · P(Bin(G, p) ≥ G·S/B) = α, with the log-linear interpolation. The observed S stands in for the unobserved W̄.

**Not licensed:** empirical Bernstein and betting. They need the W_g themselves, which are not observed. ∎

**Exact computation.** For MAE and pinball, the truncated objective a ↦ Σ min(ρ(y_i − a), κ) is piecewise linear, with kinks at y_i, y_i − κ/q and y_i + κ/(1 − q). Pythia evaluates it exactly at every kink, using prefix sums. Window-miss is computed by an exact sliding window. Any *approximate* minimizer could exceed the value at a\* and would invalidate the certificate, so none is used.

## Why it is tighter

| Source of looseness | Block-SAA (k = 64) | Dominated full-sample |
|---|---|---|
| Optimism bias | per block (≈ 8 patients per cell per block) | per full cell (≈ 512 patients), negligible |
| Concentration units | G/k = 64 block scores, betting | G = 4,096 clusters, Bentkus |
| Variance adaptivity | yes (dKelly) | mean-adaptive only (Bentkus binomial comparison) |
| Outcome support | not needed (score truncation) | not needed (loss truncation) |

**Fragmentation.** With finer cells, concentration is unchanged: it always runs over all G clusters. Only optimism grows, and a singleton cell contributes zero. Fragmentation cost is therefore purely optimism. Section 2 of the plan quantifies it and pairs it with the declared second-order pooling penalty.

## Relation to the literature

- **E[min SAA] ≤ min E and candidate domination** are classical in stochastic programming:
  - Mak, Morton and Wood (1999) give lower bounds and optimality-gap estimators;
  - Bayraksan and Morton (2006) assess solution quality.

  Those works use batch-means or CLT intervals.
- **Bentkus (2004)** and **Hoeffding (1963)** are classical.
- **Candidate contribution:** a finite-sample, support-free, cluster-valid Bayes-floor certificate over representation cells. It comes from range-only, one-sided inversion of an *unobserved* dominating sum, with exact truncated optimization. It covers MAE, every pinball level and window-miss, and plugs into the lattice and the fixed-policy value bridge.

  This is a composition result, not a new inequality.

## Limits

- It is not variance-adaptive. A weak-interaction Bernstein (Maurer 2017/2018) remains the route to variance sensitivity.
- Bentkus with a common range pays for unequal cluster sizes. Outcome-blind retention (`max_cases_per_cluster`) caps the range at a safe cost.
- Heavy contamination needs larger caps, and the cap grid is part of the alpha ledger.
