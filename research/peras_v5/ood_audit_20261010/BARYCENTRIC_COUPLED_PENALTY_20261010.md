# Barycentric coupled penalty: a sharper loss-specific near-replicate correction

## What this addresses
`COUPLED_ENVIRONMENT_TRANSPORT_20261010.md` closed with two open questions:
1. sample efficiency;
2. whether pinball and MSE can use sharper loss-specific discrepancies than the generic Lipschitz × W1 penalty.

With that penalty, Q90 and MSE gave **0/50** positive certificates. MSE pays 2·D·W1, which is 600 × W1 on a [0, 300] clip.

This note answers question 2. It also completes the planned sample-split alignment route (`run_learned_alignment_environment_ood.py`), which had been written but not yet run.

## Theorem (barycentric penalty)
**Setting**
- Blocks are disjoint and fixed without outcomes.
- Unit laws Q_j are clipped, and optionally aligned by a shift frozen without evaluation or cap outcomes.
- Evaluation outcomes are independent across units.
- Inside a block, (V_1..V_k) is **any** coupling of (Q_1..Q_k), and V̄ is its mean.
- S is the in-tuple SAA score, and T = k⁻¹ Σ_j R(Q_j).

**Claims**
- MAE: E S ≤ T + E[k⁻¹ Σ_j |V_j − V̄|].
- Every pinball level q, including Q90: E S ≤ T + ½·E[k⁻¹ Σ_j |V_j − V̄|].
- MSE: E S ≤ T + E[k⁻¹ Σ_j (V_j − V̄)²].

**Proof sketch** (full proof in `barycentric_coupled_bound.py`)
1. **Start from SAA optimism.** E S ≤ inf_a k⁻¹ Σ_j E ℓ(Y_j − a). The right-hand side depends only on the marginals, so the evaluation outcomes can be replaced by the coupled V_j.
2. **Pinball and MAE.** The asymmetric subgradient bound gives ρ_q(V_j − a) ≤ ρ_q(V̄ − a) + q·d_j⁺ + (1−q)·d_j⁻, with d_j = V_j − V̄.
   - Because Σ_j d_j = 0, the block means of d⁺ and d⁻ are both ½·mean|d|. This gives the multiplier ½ for **every** q, instead of the generic max(q, 1−q) = 0.9.
   - MAE = 2ρ_½, which gives multiplier 1.
3. **Back to T.** Jensen applied to ℓ(V̄ − ā) = ℓ(mean_j(V_j − a_j*)) shows R(law V̄) ≤ T.
4. **MSE.** The identity k⁻¹ Σ(V_j − a)² = (V̄ − a)² + k⁻¹ Σ(V_j − V̄)² holds exactly. Then Var(V̄) ≤ (mean_j sd_j)² ≤ T. ∎

**Comparison with the pairwise route**
- Since |V_j − V̄| ≤ k⁻¹ Σ_i |V_j − V_i|, the absolute penalty never exceeds the pairwise cost.
- For MSE, the penalty becomes a squared coupled dispersion instead of 2D·W1. This is the decisive change.

**Calibration and novelty**
- The penalties are environment means of bounded statistics (ranges D/2 and D²/4), calibrated exactly as before: minimum of Hoeffding, empirical Bernstein and Bentkus, with the budget split over both penalty families.
- The ingredients (SAA, Jensen, couplings) are classical. The contribution is the loss-specific barycentric form and its outcome-only calibration.

## Known-law check (`check_barycentric_penalty.py`)
**Setup**
- 120 random blocks: k ∈ {2, 3, 4, 8}; normal, Laplace or Gamma(2) shapes; random locations and scales.
- Comonotone couplings, with 10⁵ Monte Carlo tuples.
- Exact Bayes risks by quadrature.

**Results**
- **0/360 violations.**
- Median (E S − penalty)/T, the tightness before any concentration loss:

| Loss | Barycentric | Pairwise |
|---|---|---|
| MAE | 0.226 | 0.053 |
| MSE | 0.647 | (pairwise is hopeless here) |
| Q90 | −1.01 | −2.93 |

Q90 remains limited by the optimism of the small-k in-tuple witness, not by the penalty.

## Frozen fresh OOD audit (`run_barycentric_ood.py`, `barycentric_ood_results.jsonl`)
**Pre-registered before running**
- Seeds 103001–103005; designs 1–10; n = 800.
- 5,000 environments per coupling.
- k ∈ {2, 4, 8}; the existing score clips; outcome clip [0, 300].
- Aligned routes: per-unit median from the first half of the environments, cap from the second half, residual clip [−150, 150]. This is the design of `run_learned_alignment_environment_ood.py`.
- Budgets are identical to the earlier audit. Each route is a separate 95% procedure.

Each cell gives positive bounds out of 50, mechanisms with any positive bound out of 10, the median ratio of bound to original gold, and violations:

| Route (common-quantile environments) | MAE | Q90 | MSE |
|---|---|---|---|
| Pairwise, unaligned (previous) | 46/50 · 10 · 0.143 · 0 | 0/50 · 0 · 0 · 0 | 0/50 · 0 · 0 · 0 |
| **Barycentric, unaligned** | 46/50 · 10 · 0.140 · 0 | 5/50 · 1 · 0 · 0 | **39/50 · 9 · 0.036 · 0** |
| Pairwise, aligned (the planned route) | 50/50 · 10 · 0.536 · 0 | 30/50 · 6 · 0.200 · 0 | 0/50 · 0 · 0 · 0 |
| **Barycentric, aligned** | **50/50 · 10 · 0.550 · 0** | **38/50 · 8 · 0.371 · 0** | **50/50 · 10 · 0.315 · 0** |
| Any route with product coupling | 0/50 | 0/50 | 0/50 |

**Paired comparison, aligned, barycentric vs pairwise** (out of 50 datasets):

| Loss | Barycentric strictly tighter | Barycentric looser |
|---|---|---|
| MSE | 50 | 0 |
| Q90 | 38 | 0 |
| MAE | 40 | 10 |

**Unaligned:**
- MSE tighter in 39/50 and Q90 tighter in 5/50.
- MAE is looser in 40/50. At the selected k = 2 the two penalties coincide, and the barycentric route spends half its cap budget on the squared family. A route registered for MAE alone would remove this cost.

**Totals:** across all 8 routes × 3 losses × 50 datasets there are 0 violations against the original-outcome gold.

## Population check (`check_barycentric_population.py`)
- All 600 caps (50 datasets × 3 k × 2 routes × 2 penalty families) were compared with the population comonotone dispersions by 5,000-point quadrature.
- **0 misses.**
- Minimum margins: 0.45 for the absolute penalty, 39 min² for the squared one.

## Interpretation
1. **Where the gains come from.** The barycentric penalty turns MSE from impossible to routine: 0/50 → 50/50 aligned, median 0.32 of gold. It lifts Q90 to 38/50 and median 0.37.
2. **Location alignment matters most for MAE.** Aligning raises MAE tightness from 0.14 to 0.55, because Bayes risk under a location loss ignores per-unit shifts, while the transport penalty does not.
3. **Coupling design is still indispensable.** Product coupling certifies nothing on any route. The contract is unchanged: repeated environments with registered coupling, alignment frozen without evaluation outcomes. **No Mayo number or claim tier changes.**

## Next steps
- **Alignment without environments.** The shift may be any function fixed without evaluation and cap outcomes. A shift from an independent observational split (a cross-fitted location map) keeps T unchanged by translation invariance. It would halve the environment requirement, since the environments are then used only for the cap. The cost is a learned (not oracle-median) shift.
- **Q90's remaining bottleneck is the witness.** The weak part is the small-k in-tuple witness, not the penalty. The multi-point witness from `FINITE_GROUP_MULTILOSS_AUDIT` should be combined with the barycentric penalty under one alpha ledger.
- **A single-loss MAE route.** Register a MAE-only route that does not split its cap budget, to remove the unaligned MAE cost.
