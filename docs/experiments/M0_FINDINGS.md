# M0 — Bregman core: acceptance report

Run: `python scripts/run_m0_suite.py` (100 synthetic designs, R\* known by
construction, ~3 min). Raw output in `outputs/m0_suite.json`.

## Acceptance criteria and where they landed

| criterion | target | regression (MSE) | classification (logloss) |
|---|---|---|---|
| validity `B ≤ R*` | 100% | **88.3%** (53/60) | **100%** (40/40) |
| median \|Ê − R\*\|/R\* | < 5% | **12.6%** | **4.7%** |
| usable tightness | — | 0.548 | 0.827 |
| Ê within 5% | — | 13/60 | 20/40 |

**Verdict: the cross-entropy branch passes; the squared-error branch does not.**
Log loss is deployable as it stands. MSE needs R1 before it can carry a client
claim.

## What went wrong, precisely

All seven failures are structural, not random. Validity by mean-function shape:

| shape | n | validity |
|---|---|---|
| linear | 12 | 0.92 |
| nonlinear | 14 | 1.00 |
| step | 17 | 1.00 |
| **interaction** | **17** | **0.65** |

Six of seven failures are `interaction` designs, whose mean function is
`2·x₀·x₁ + 1.5·tanh(x₂) + …`. Every learner in the pool — boosting, forest,
ridge, kNN — under-fits a pure product term in the *same direction*. That shared
approximation bias is invisible to the disagreement statistic: the models agree
with each other while all being far from `m`, so `Δ̂` under-states the true excess
risk and `B = L − Δ̂` lands above `R*`.

Worked example, `reg-0006`:

```
R*        0.5000        true Δ_k = L − R*  =  0.3700
L         0.8700        max pairwise Δ̂     =  0.2937   ← under-states by 21%
B         0.5762  >  R*                             ← invalid by 15.2%
```

Decomposing the largest pair (hgb vs ridge): under orthogonality
`E[(f₁−f₂)²]` should be `Δ_hgb + Δ_ridge = 0.37 + 0.89 = 1.26`; the observed
value is `0.59`. The cross term `E[(f₁−m)(f₂−m)] ≈ 0.34` is the shared bias.

**This is the A-fold assumption failing, and it is not repairable inside the
Bregman identity.** Two identical models have zero disagreement and arbitrary
excess risk, so no assumption-free upper bound on `Δ` exists from disagreement
alone. The honest options are to estimate the dependence (R1), or to accept a
stated assumption and report it.

## Secondary findings

**Validity degrades at low SNR.** snr = 0.5 → 0.73 validity; snr ≥ 1.0 → 0.92–0.94.
At low SNR the excess risk is a small fraction of `L`, so the same absolute error
in `Δ̂` is far more likely to push `B` above `R*`.

**Tightness degrades at high SNR.** snr = 4.0 → usable tightness 0.239 versus
0.755 at snr = 1.0. When the floor is small relative to model error, the bound is
valid but nearly vacuous. This mirrors the matching-based comparators' known
failure at high SNR, and is worth reporting as a regime map rather than a single
number.

**Finite-sample noise in L is real.** 10 of 100 designs have empirical `L` *below*
the population `R*`. At n = 800 with `R* ≈ 1`, the standard error of the OOF risk
is ≈ 0.05, so the comparison `B ≤ R*` is itself noisy. Any coverage statement
(R3) must account for this, not only for the envelope.

## The `(Δ, λ)` identification attempt — negative result

`pythia/core/folddep.py` implements an idea that looked promising and does not yet
work. Because every predictor's risk is `R* + Δ_i`, **risk differences are excess
differences and are observable exactly**:

```
Δ_i − Δ_k = Risk_i − Risk_k =: d_i
```

Positing one shared dependence `E[(f_i−m)(f_j−m)] = λ√(Δ_iΔ_j)` gives

```
p_ij = Δ + (d_i + d_j)/2 − λ√((Δ + d_i)(Δ + d_j))
```

— `B(B−1)/2` equations in two unknowns, so `(Δ, λ)` is over-determined and, on
paper, identified from data alone. In practice the fit is only **weakly
identified**: `Δ` and `λ` trade off along a ridge. On `reg-0006` the solver
returns `Δ̂ = 0.000, λ̂ = 0.902`, and the conservative reading gives `Δ = 2.64`,
which drives `B` negative — valid but vacuous. Per-pair implied `λ` ranges over
0.58–0.93 on that design, so the single-`λ` model is also mis-specified.

The module is shipped, marked experimental, and reported in
`diagnostics.lambda_hat`. **It is never used to compute `B`.** Fixing the
identification is R1's first task; the observation that risk differences pin
`Δ_i − Δ_j` exactly is worth keeping either way.

## Three alternative Δ estimators — all worse

Before settling, four constructions of `Δ̂` were run head-to-head on the same 100
designs. `d_j = Risk_j − L` is observable exactly, and under orthogonality
`p_kj = Δ_k + d_j/2`, so the "anchored" variants look like exact identifications
of `Δ_k`. They are not robust:

| Δ̂ construction | valid (all) | reg | clf | usable tightness |
|---|---|---|---|---|
| **`maxpair` — max over all pairs** | **93/100** | **53/60** | **40/40** | **0.660** |
| `anch_max` — max_j (p_kj − d_j/2) | 23/100 | 7/60 | 16/40 | 0.214 |
| `anch_med` — median_j (p_kj − d_j/2) | 12/100 | 6/60 | 6/40 | 0.114 |
| `k_min` — min over pairs involving the best | 31/100 | 17/60 | 14/40 | 0.289 |

The maximum over *all* pairs is not a crude choice — it is absorbing the
dependence the identity ignores. The anchored variants are exactly right under
orthogonality and collapse the moment orthogonality fails, which is most of the
time. **Keep `maxpair`.**

The price is `pool_health`: because the bound uses the largest disagreement, one
predictor far off the pace inflates `Δ̂` and drives `B` to zero. A `pool_health`
warning fires when the worst predictor's risk exceeds twice the best's. Pools
must be competitive *and* heterogeneous — a real constraint on S3, now recorded
in the certificate.

## What ships

- `B` uses the **orthogonal rung**: `Δ̂ = max` over pairwise disagreement.
  Measured validity 93/100 overall — reported, not claimed.
- `Ê` uses the **mean** pairwise disagreement. Systematically conservative
  (median signed error −5.6%), because the mean over pairs over-states the *best*
  predictor's excess.
- Every certificate carries `assumptions: ["A-fold"]`, `coverage_guarantee: null`
  and the `lambda_hat` diagnostic.

## What this changes in the plan

1. **Promote R1.** It was scheduled as the fix for the guarded branch's rung
   ambiguity. M0 shows the *exact* branch needs it too. R1 is now the gating
   research item for both branches.
2. **Ship logloss first.** The cross-entropy branch met both criteria. That is
   the demo, and it is also the branch that touches the scaling-law question.
3. **Add a regime map to the certificate.** Validity and tightness both depend
   strongly on SNR and on mean-function structure. A certificate should say which
   regime the data sits in, not only what the number is.
4. **R3 must cover sampling noise in L**, not only envelope calibration.

## Reproducing

```bash
python scripts/run_m0_suite.py --quick     # 14 designs, ~30 s
python scripts/run_m0_suite.py             # 100 designs, ~3 min
python -m pytest tests/ -q                 # 7 unit tests, < 1 s
```

Every design is a pure function of its integer seed, so the suite regenerates
bit-identically.
