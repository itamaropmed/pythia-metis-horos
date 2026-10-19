# Local-risk certificate v2 — theorem and pipeline

Setting: deployed model f fitted on FIT entities only; CAL disjoint; loss ℓ ∈ [0, M]; hierarchy H from f's CAL
predictions + fixed covariates (bin edges from CAL predictions only); support partition P from H and CAL entity
ids (finest-first residual rule, threshold m), reading no outcomes. R = regions with ≥ m CAL entities, plus global.

Estimands: θ_r = case-weighted expected loss of f over CAL cases in r; Δ_r = θ_r − θ_global.

Claim: with probability ≥ 1 − α, simultaneously for every model and r ∈ R: θ_r ∈ [L_r, U_r] and Δ_r ∈ [Lᶜ_r, Uᶜ_r],
each endpoint a heterogeneous-support Chernoff inversion at α / (2 · #models · #tests) with entity-cluster sums.

Why: intervals are valid one-sided bounds for sums of independent bounded entity contributions; the test family
is fixed before outcomes (P, R, contrasts depend only on predictions, covariates, ids); union bound; a new case's
region is a function of its covariates and f, so routing is selection-free.

Why v2 over v1 (OOF route): v1 certified fold models but deployed a refit, and gave backoff cases the interval of a
coarser node whose population differs. Simulation: v1 5.6% wrong certified rows (19/19 replicates); v2 0.

Λ transport: odds(r′) ∈ [odds(L)/Λ, Λ·odds(U)]. Tipping Λ\* = smallest Λ at which the certified direction can fail.
Simulated median Λ\* ≈ 1.1; Mayo median 1.42–1.55. Report Λ\* with every direction.

Planning m: the threshold is outcome-blind, so choose it before outcomes from a width tolerance. Larger m → fewer,
wider-population regions with narrower intervals (simulated certified share 0.06 at m=100, 0.46 at m=400).
