# Pythia local-risk certificate: theorem, claims backbone and pipeline (v2, 2026-10-18)

This document supersedes the OOF-based route in `scripts/run_mayo_pythia_local_w30.py` for any claim above `exploratory`.

- **Evidence:** `docs/experiments/PYTHIA_LOCAL_RISK_V2_SIMULATIONS_20261018.md`.
- **Theory detail:** `docs/theory/LOCAL_RISK_V2_SPLIT_PARTITION_TRANSPORT_20261018.md`.

## Theorem (v2)

**Setting.**
- A deployed model f is fitted on FIT patients only.
- CAL is a disjoint set of calibration patients.
- ℓ ∈ [0, M] is an attested bounded loss; W30 miss has M = 1.
- H is a finite hierarchy built from f's predictions on CAL plus fixed covariates. Its bin edges come from CAL predictions only.
- P is the support partition built from H and the CAL patient identifiers (finest-first residual rule, threshold m). P reads no outcomes.
- For each model, R is the set of regions of P with at least m CAL patients, plus the global node.

**Estimands.** For each model, region r and the global node g:
- θ_r = case-weighted expected loss of f over the CAL cases in r;
- Δ_r = θ_r − θ_g.

**Assumptions.**
- (A1) Given FIT and the CAL design, outcomes of different CAL patients are independent. Repeated cases of one patient may depend arbitrarily on each other.
- (A2) ℓ ≤ M almost surely.

**Claim.** With probability at least 1 − α, all of the following hold at once:

  θ_r ∈ [L_r, U_r] and Δ_r ∈ [Lᶜ_r, Uᶜ_r]  for every model and every r ∈ R,

where each endpoint is a heterogeneous-support Chernoff inversion at level α / (2 · #models · #tests), with patient-cluster contributions.

**Proof sketch.**
1. Under (A1)–(A2) every interval is a valid one-sided bound for a sum of independent, bounded patient contributions.
2. The family of tests is fixed before outcomes are read: P, R and the contrasts depend only on predictions, covariates and identifiers.
3. A union bound over the family gives simultaneous coverage.
4. A new case's region is a function of its covariates and f, so routing is selection-free.

**Corollary (categories).**
- `certified_higher_than_global` ⇔ Lᶜ_r > 0.
- `certified_lower_than_global` ⇔ Uᶜ_r < 0.
- Otherwise `local_shift_unresolved`.
- Cases that fall through to the global node get `global_only`.
- Failed attestations give `refused`.

On the simultaneous event, no certified direction is wrong for the calibration population of the region the case was routed into.

## Claims backbone (what may be said, at which tier)

| # | Claim | Tier | Needs |
|---|---|---|---|
| C1 | Region interval and direction for the CAL population of the routed region | `conditional` | A1, A2 |
| C2 | The same interval for the *later* population of that region | `conditional on Λ` | A1, A2, plus a declared within-region odds-shift budget Λ. The transported interval is odds(r′) ∈ [odds(L)/Λ, Λ·odds(U)]. Each certified direction also reports its tipping value Λ\*. |
| C3 | "This region's later risk exceeds its certified upper bound" | anytime-valid test (level α per region) | Later losses conditionally independent given the past (first case per patient). |
| C4 | Development-to-later agreement counts | descriptive | none: reported, never used for selection |
| — | Individual-case outcome guarantee | **not claimed** | — |
| — | Correlation between risk and error | descriptive only, no threshold | — |

**What changed from v1, and why**

| v1 (current Mayo replay) | v2 | Simulation evidence |
|---|---|---|
| OOF fold-model losses; deployed model refitted | Certify the deployed model on separate CAL patients | v1 gave wrong directions in 19/19 replicates (5.6% of certified rows); v2 gave 0 |
| A case is given its backoff node's interval | Support partition: the certified estimand *is* the routed population | Routed-population misses: 160/2,046 cells (v1) vs global fallback only (v2) |
| α charged to every node, including unusable ones | Eligible-node ledger | About 4× fewer tests, narrower intervals |
| Direction requires two disjoint intervals | Direct contrast interval | About 2× more certified cases with no errors |
| No shift statement | Λ transport, tipping Λ\*, harmful-shift monitor | Coverage under 40% within-cell noise drift: 0.88 at Λ = 1, 0.99 at Λ = 1.25 |

## Pipeline

```text
development (before cutoff)                                     later cohort (after cutoff, new patients)
 ├─ FIT patients (earliest first case, 60%) ── Horos fits f_A, f_B ──────────────┐
 └─ CAL patients (remaining 40%)                                                 │
      predictions f(X_cal) ── bin edges ── hierarchy H ── support partition P(m) │  (outcome-blind)
      losses 1{|y − f| > 30}                                                     │
      Pythia: eligible ledger + region intervals + contrasts (one α ledger)      │
                                   │                                             ▼
                                   └──────── frozen registry ──► route each later case ► case card
                                                                 (frozen BEFORE outcomes)
case card: model, region, θ estimate, [L, U], relation, Λ* (if certified), Λ-transported interval
audit (after outcomes): later W30 per region, coverage at Λ ∈ {1, 1.25, 1.5}, monitor alarms
```

**Implementation**
- Pythia: `pythia/core/local_model_risk.py` (`build_support_partition`, `alpha_ledger="eligible"`, `relation_test="contrast"`) and `pythia/core/local_transport.py`.
- Mayo replay: `scripts/run_mayo_pythia_local_w30_v2.py`, which outputs aggregates only and was checked end to end on synthetic Mayo-shaped data.

**Declared before outcomes:**
- the fit share (0.6);
- the primary threshold m = 100, with 200 and 400 reported as predeclared sensitivity analyses;
- α = 0.05;
- Λ grid {1, 1.25, 1.5}.

## Status of the Mayo claims

- The previously reported 43–44% certified directions came from the v1 route. Treat them as **exploratory** until the v2 script is run locally.
- In simulation, v2 certified fewer cases at m = 100. Its main lever is the support threshold, which must be fixed before outcomes are seen; taking the best of 100/200/400 afterwards is not allowed.
