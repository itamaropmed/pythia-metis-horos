# Smooth domination: tighter Bayes-floor certificates for continuous information

## Why it is needed (lessons from the Codex suites)

Codex's continuous-feature suites showed four failures:

| Codex finding | What it means for the bound |
|---|---|
| Second ten-design MAE challenge: pair/group envelope median tightness 0.486; sparse 12D gave 0 | Near-replicate penalties are first order (Lipschitz × distance), so blocks must stay tiny |
| Multi-loss suite: paired Bentkus beat the adaptive group route on 543/800 pinball cells | Larger groups lost because their drift penalty grew linearly with group diameter |
| kNN estimator beat the witness centre in 190/200 runs | Estimator issue, not a bound issue (out of scope here) |
| Only an oracle projection rescued sparse geometry | Metric dilution |

The common root is a **first-order drift penalty paired with block-level concentration**. That forces pairs, and pairs pay the largest optimism: for Gaussian MAE the k = 2 witness is only 0.71 of the floor, and for Q90 only 0.32.

## Theorem (second-order smooth domination)

**Setting**
- Rows are independent; covariates x_j are fixed.
- The Bayes action a\*(x) (the median, or the q-quantile) is L-Lipschitz. **Declared.**
- The conditional density near the action is at most f_max / s(x). **Declared.**
- Curvature constants: κ_j = 2 f_max / s_j for MAE and f_max / s_j for pinball; slope constant c_q = max(q, 1 − q).
- Rows are partitioned without looking at outcomes into blocks b with centroids c_b, and d_j = ‖x_j − c_b‖.

**Quantities**
- Observable statistic: S = (1/n) Σ_b min_a Σ_{j ∈ b} min(ℓ(Y_j − a), τ), computed exactly.
- Unobservable comparison terms: W_j = min(ℓ(Y_j − a\*(c_b)), τ).

**Claims**
1. **Pathwise domination.** S ≤ mean_j W_j.
2. **Each comparison term is close to its own floor.** E W_j ≤ R_j + pen_j, with

   pen_j = min( c_q L d_j , (κ_j/2) L² d_j² ).

   This holds because the expected loss is convex in a, with slope at most c_q and curvature at most κ_j.
3. **Lower endpoint.** R ≥ LCB(S) − (1/n) Σ_j pen_j, where the LCB is either:
   - Bentkus over all n rows, which is valid by claim 1; or
   - dKelly over the m = n/k independent block scores, which is variance-adaptive.

   The penalty depends on the features only.

**Proof.** Claim 1 holds because a minimum over a is at most the value at the fixed a\*(c_b). For claim 2, take the second-order Taylor expansion of a ↦ E ℓ(Y_j − a) around a\*(x_j), where the derivative is zero. The first-order bound comes from the slope. Truncation can only lower W_j. Claim 3 follows by combining claims 1 and 2 with the chosen concentration inequality. ∎

**Why it is tighter**
- The drift penalty scales as L² d² instead of L·d. Blocks of 16–64 rows remain affordable, and in-block optimism vanishes: the MAE ratio is 0.98 at k = 32.
- Domination lets Bentkus use all n rows. The block route keeps dKelly's variance adaptivity.
- Both routes enter one Bonferroni ledger, so the portfolio is never much worse than the pair route it contains (k = 2).

**Same declarations as the accepted methods.** The comparison routes use the same constants: L, which every near-replicate or kNN lower bound needs, and f_max, which the curvature theorem already uses. Neither is identified from the data. Both are reported with tipping points.

## Algorithm (PYTHIA-S)

1. Choose outcome-blind geometries: a kd-tree (best in low dimension) and recursive greedy nearest-neighbour matching (best as dimension grows). Use block sizes 2, 4, …, 64.
2. Compute the exact truncated SAA per block over a cap grid.
3. Compute the penalty for each block geometry from the features.
4. Take the maximum over geometry × k × cap × {domination-Bentkus, block-dKelly}, all Bonferroni-charged.
5. Report B(L) and the tipping L\*.

## Known limits

- **Higher dimension with smooth but non-constant actions** (linear in 5D): block diameters stay large. The candidate next theorem is a local-linear witness, which makes the penalty depend on the Hessian, so linear signals cost nothing. Its exact truncated optimization in d + 1 parameters is the open computational issue.
- **Sparse designs** need a declared metric or an outcome-blind learned key whose Lipschitz constant is declared in key space.
- L and f_max are declared. Misdeclaring them breaks validity, exactly as it does for every accepted near-replicate method.
