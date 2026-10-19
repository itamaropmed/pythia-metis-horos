# Pythia diagnostics on known-signal / known-noise data (2026-10-14)

## Question

Where exactly does a training-free Pythia floor certificate lose tightness, and what is it missing?

## Setup

**Model.** Y = f(X) + s(X)·ε, with exactly computable Bayes floors for MAE, Q90 and MSE. The target is the fixed-design floor of the certified units.

**Ten mechanisms:**
- linear (d = 5);
- Friedman-type (d = 5);
- sparse (2 signal and 18 noise features);
- high-frequency (d = 2);
- heteroscedastic;
- t3 noise;
- lognormal (skew) noise;
- discrete exact cells;
- high SNR;
- low SNR.

**Protocol**
- Split: half A is the learning half (outcomes allowed); half B is certification.
- Concentration: Bentkus and diversified-Kelly lower confidence bounds (α split between them).
- Penalties are evaluated at the **true** discrepancy. These are oracle-discrepancy checks: they isolate every other source of looseness. The declaration problem is discussed separately in section 6.

**Runs and validity**
- Round 1: 10 mechanisms × 8 seeds, n = 4,000 (2,000 certified units).
- Round 2: 10 × 10 seeds at n = 4,000, plus 4 × 3 seeds at n = 16,000.
- Validity: **0 violations in 13,592 certificate rows.** An independent Monte-Carlo check of the new inequality (240 random heterogeneous blocks; Gaussian, t3 and lognormal noise; MAE and Q90) never exceeded its bound. The maximum standardized excess was −22.

## 1. Decomposition of the current route: the penalty is the problem

Medians over all mechanisms and seeds, as a fraction of the true floor:

| Route (k = 8) | Exact-replicate witness (optimism) | Mean witness | Penalty at true P | Width | Bound | Tipping |
|---|---|---|---|---|---|---|
| raw kd, MAE | 0.91 | 1.23 | **0.84** | 0.09 | 0.33 | 1.4 |
| learned key, MAE | 0.91 | 1.01 | **0.45** | 0.06 | 0.49 | 2.1 |
| aligned, MAE | 0.91 | 0.99 | **0.41** | 0.06 | 0.52 | 2.2 |
| learned key, Q90 | 0.81 | 0.88 | **0.95** | 0.06 | **0.00** | 0.9 |
| learned key, MSE (unbiased witness) | 1.00 | 1.18 | 0.19 | 0.14 | 0.84 | 5.1 |
| oracle key (blocks on the true law), MAE | 0.91 | 0.91 | 0.02 | 0.06 | 0.83 | 53 |

**Readings**
- Concentration width is small: 0.06–0.14. dKelly beat Bentkus in 100% of rows.
- Witness optimism is modest for k ≥ 8.
- **The penalty dominates.** For MAE, the witness is actually inflated by only about 0.10 (1.01 − 0.91), but the barycentric penalty charges 0.45: **about 4.5× too much.** For Q90 the penalty alone wipes out the certificate.
- MSE does well because its penalty, k/(k−1)·Var(means), is already **second order** and exact.
- MAE and pinball penalties are **first order**, using a worst-case coupling.

## 2. The missing key: a second-order (curvature) penalty

**Theorem (curvature / Jensen-gap penalty).** For any block, with *any* dependence inside the block:

E[witness_b] ≤ R(mixture_b) ≤ T_b + min( c_q · mean_j |a\*_j − ā|, (κ/2) · mean_j (a\*_j − ā)² )

where:
- a\*_j is unit j's Bayes action (its median or quantile) and ā is their block mean;
- c_q = max(q, 1 − q), so c = 1 for MAE;
- κ is a bound on the curvature of a ↦ E ℓ(Y_j − a), which is 2·(sup density) for MAE and sup density for pinball.

The same formula with κ = 2 recovers the MSE penalty.

**Proof.** E inf ≤ inf E gives the first inequality. Each expected loss is convex with second derivative at most κ, so E ℓ(Y_j − ā) ≤ R_j + (κ/2)(ā − a\*_j)². ∎

The exact quantity being bounded is the **Jensen gap of the Bayes risk over the block mixture**. That is the loss-specific information between "which unit" and the outcome inside a block. The penalty is therefore the leftover signal inside a block, measured in the loss's own geometry.

## 3. Results

n = 4,000. Each value is the median over mechanisms of the per-mechanism median, at the best k.

| Loss | Configuration | Current (barycentric) | **Curvature** | Exact Jensen gap (ceiling) | Oracle key |
|---|---|---|---|---|---|
| MAE | learned key | 0.50 | **0.85** | 0.85 | 0.85 |
| MAE | aligned (raw blocks) | 0.52 | **0.85** | 0.85 | — |
| MAE | raw kd | 0.37 | 0.71 | 0.83 | — |
| Q90 | learned key | **0.00** | **0.68** | 0.83 | 0.74 |
| Q90 | aligned | 0.00 | **0.69** | 0.83 | — |

**Scaling to n = 16,000** (linear, Friedman, t3, sparse):
- MAE with curvature: 0.91–0.95 (barycentric: 0.64–0.73).
- Q90 with curvature: 0.77–0.89 (barycentric: 0.25–0.40).

**Robustness.** The tipping margin in drift-standard-deviation units rises from about 2.1–2.3 to about 2.9–3.2. A curvature-penalty variance tip is square-rooted for this comparison. So the bound is both **tighter and more robust** to a misdeclared discrepancy.

## 4. Other weaknesses found

1. **Curvature constant.** The global sup-density κ is conservative for skew and heteroscedastic Q90 (skew: 0.12 versus 0.77 with the exact Jensen gap; heteroscedastic: 0.00 versus 0.81). A **local curvature**, i.e. a density bound near the quantile, should close most of this gap.
2. **Geometry.** Raw Euclidean blocks fail in sparse and high-SNR settings (MAE 0.07 and 0.00 with the current penalty). Pythia should always block on an **outcome-blind key learned on an independent split**, and/or align by a frozen location map.
3. **Heavy tails.** For MSE with heavy tails (t3, lognormal), even the oracle reaches only 0.62–0.70. Concentration of heavy-tailed scores needs better truncation or cap design.
4. **Local-linear witnesses.** Fitting a + b·z inside a block removes linear drift. This helps only raw, unlearned geometry (MAE 0.75, Q90 0.65). With a learned key plus the curvature penalty it is unnecessary.
5. **Point estimators** (estimate tier). Nearest-neighbour residual variance overshoots badly in high-dimensional or high-SNR settings (2.9× and 4.1× the true floor). A learned key fixes the sparse case (1.1×) but not high SNR (2.3×).

## 5. What Pythia lacks, ranked

1. **A second-order penalty for MAE and pinball.** This is the main missing key. It turns MAE from about 0.5 into about 0.85 and Q90 from 0 into about 0.7 of the floor at the true discrepancy.
2. **Local rather than global curvature constants.**
3. **A learned, outcome-blind key or alignment from an independent split as the default geometry.**
4. Heavy-tail concentration.

## 6. What this does *not* solve

- The within-block dispersion of Bayes actions is still **not identifiable** from one draw per unit. It must still be declared, calibrated or falsified.
- What changes is that the declaration is now **second order**: a variance of action drift, which is roughly the learner's error variance inside a block. That makes the certificate more tolerant of a misdeclared value.
- The new route also needs a declared density (curvature) bound.
- All results are synthetic. **No Mayo number changes.**
- Round 2 chose the score cap from the certification half. That is a minor optimism, and production must take it from the learning half; round 1 did so.

## 7. Next steps

1. Implement the curvature penalty in `pythia.core.declared_modulus` as a new penalty family, with declared (V, κ) and provenance. Pre-register a development run, then an untouched confirmation, on the frozen OOD designs.
2. Local-curvature refinement: a declared density bound near the quantile, aiming to reach the Jensen-gap ceiling (about 0.83 for Q90).
3. Non-oracle declaration experiment: calibrate V by twin plus the m₉₅ audit, and use the falsification lower limit. Measure coverage and tightness without oracle discrepancy.

## 8. Subsequent frozen confirmation

The second-order route was promoted into `pythia.core.declared_modulus`, with a
fail-closed local-curvature domain and optimized heterogeneous-curvature
geometry.  A fresh run fixed the score cap entirely on the learning half.  On
400 untouched loss/configuration rows it had zero known-floor exceedances.
MAE improved on 10/10 mechanism medians for both tested geometries, raising
median tightness from about 0.50--0.52 to 0.84.  Aligned Q90 improved on 10/10
mechanism medians and rose from zero to 0.63 median tightness; learned-key Q90
won 8/10 (`p=0.0547`).  Q90 remained zero for high SNR and weak under
heteroscedasticity and skew.  Full protocol and raw hash are in
`CURVATURE_CONFIRMATION_RESULTS_20261014.md`.

**Files:**
- `sim.py`: round 1 (strategy ladder);
- `sim_ll.py`: local-linear witness;
- `sim2.py`: penalty geometry;
- `report.py` and `report2.py`.
