# LP bounds with a relaxation dial, and the conditional agreement geometry of text

Code: `pythia/core/lp_bounds.py`.
Tests: `tests/test_lp_bounds.py`.
Experiments: `research/peras_v5/lp_20261017/`.

## 1. Why an LP

Every certificate built so far is one fixed inequality. Partial identification instead asks a single question: **what is the smallest best-achievable risk consistent with everything that has been certified?** "Everything" means every certified design statistic, together with the exact structural facts and the declared assumptions. That question is a linear program. Its dual weights show which constraints bind.

**Choice of target.**
- **Prediction tasks** use the Brier floor G = mean(1 − |p|²). The 0-1 floor follows by f_K(G) ≤ R₀₁ ≤ G, and the log floor by H ≥ G.
- **Estimation tasks** use the MSE floor, mean σ², through ½(Y − Y′)². The identity is the same.

## 2. LP-1: the design LP

**Variables**
- G: the joint floor.
- V = I_A(Y; T | X): text information beyond the table.
- D_ℓ: within-cell text drifts at nested scales ℓ.
- F_ℓ: global text drifts.

**Constraints**

| Type | Constraint | Source |
|---|---|---|
| Certified data (simultaneous boxes) | lo_ℓ ≤ G + D_ℓ ≤ hi_ℓ (within-cell text matchings); lo′_ℓ ≤ G + F_ℓ ≤ hi′_ℓ (global text matchings); gxL ≤ G + V ≤ gxU (table cells) | pair identity: every perfect matching gives G + drift |
| Exact structure | D_ℓ, F_ℓ, V ≥ 0; **D_ℓ ≤ 2V** | ½\|p_i − p_j\|² ≤ \|p_i − p̄\|² + \|p_j − p̄\|² inside a cell |
| Assumption, relaxed | D_a ≤ t_ab^γ D_b + ε and F_a ≤ t_ab^γ F_b + ε, with t_ab = (s_a/s_b)^γ | A(γ) up to slack ε |

**Outputs**
- **min G** is the certified floor L(γ, ε).
- **max G** is the assumption-free ceiling.
- **min V** is the certified text information. At ε = ∞ this is assumption-free.
- **max V** is its ceiling under A(γ).

**Relaxation path.** As ε goes from ∞ to 0, the floor moves from the assumption-free value, which is 0 by Theorem 4, to the full-assumption floor. The **tipping slack ε\*** is the smallest violation of the shape assumption that wipes out the floor. It acts as a *budget*: the certificate survives any law whose variogram departs from A(γ) by less than ε\*.

**Infeasibility.** If the LP is infeasible at ε = 0, the data refute A(γ). The LP therefore doubles as a falsification test.

## 3. LP-2: the moment LP for the 0-1 floor

Inside exact replicate cells, the falling-factorial U-statistics estimate μ_j = E_rows[p̄_c^j] without bias, for j = 1..J. Over all row-weighted distributions w of the cell law on a grid,

  min (or max) Σ_g w_g · min(p_g, 1 − p_g),  subject to μ_j ∈ box_j ± ε,

gives the **sharp** 0-1 floor implied by J certified moments. Jensen's bound f_K(G) is the J = 2 relaxation.

Higher moments come from agreement among triples and quadruples. For text-neighbour groups instead of exact cells, ε absorbs the drift, and LP-1 calibrates it through D ≤ 2V. This is the Hausdorff moment problem from the classical Bayes-error-from-moments line (for example arXiv 1105.2952), now made finite-sample.

## 4. Conditional agreement geometry of text (fibre geometry over the table)

**The induced metric.** For each table value x, the outcome law pulls back a pseudo-metric onto texts:

  d_x(t, t′)² = ½ |p(· | x, t) − p(· | x, t′)|²   (the agreement metric).

**Identifiability.**
- From single labels, *only this quadratic geometry is identifiable*, because pairs reveal ⟨p, p′⟩.
- Fisher–Rao or Hellinger geometry would need higher-order agreements, i.e. replicates or moments (LP-2).
- The observable information geometry of text is therefore the Brier/collision geometry.

**Faithfulness of a text representation φ inside fibre x.** It is measured by the within-cell variogram D_φ,x(s), the mean d_x² over φ-pairs at scale s. Its zero-lag excess is the *resolution loss* of φ.

**Per-cell observables** (`conditional_text_map`)
- **Q_cell(x):** the floor of the cell law.
- **Q_txt(x):** the disagreement among text-nearest neighbours.
- **ΔI(x) = Q_cell(x) − Q_txt(x):** text information inside fibre x, with a Holm-certified lower bound across cells.
- **resolution(x) = ΔI(x)/Q_cell(x):** the share of the cell's outcome disagreement that the text resolves.

Together these give a **map of where in the table the notes carry signal.**
