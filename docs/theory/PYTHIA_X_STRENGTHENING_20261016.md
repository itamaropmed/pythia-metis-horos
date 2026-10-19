# PYTHIA-X, part 2: the four weak points, theory and fixes

Companion to `MODULUS_FREE_EXTRAPOLATION_20261016.md`.
Code: `pythia/core/multiscale_pair_extrapolation.py`:
- `certificate_v2`
- `expectation_extrapolation`
- `increments_test`
- `replicate_anchor_v2`
- `betting_lower` / `betting_upper`

Suite: `research/peras_v5/novelty_20261016/run_strengthening.py`, `run_deep_levels.py`.

## 0. The variogram reading (organizing idea)

For a perfect matching M with lag profile s(M), E Q(M) = Ḡ + D(M). Here D(M) = mean ½|p_i − p_j|², which is the empirical **semivariogram of the label law p(x)** at that lag profile, and Ḡ is the **nugget**.

PYTHIA-X is therefore a certified, one-sided nugget estimate. The four weak points become standard geostatistical facts, now with finite-sample certificates:

| Weak point | Geostatistical reading |
|---|---|
| Adversarial failure | Micro-scale variation below the sampling lag is absorbed into the nugget (Tang, Zhang & Banerjee, JRSS-B 2021, treat the nugget as "measurement error *or* micro-scale variability"). Replicate measurements at the same location separate the two. |
| γ | The variogram's power exponent at small lags |
| Range / sill | Saturation of D at large lags, which caps the usable lag window |

## 1. Weak point 1: adversarial non-identifiability

**Proposition 1 (no assumption-free certificate without replicates).**

*Setting.*
- Partition the space into M cells of equal X-probability.
- Let p_σ(x) = ½ + δ·σ_{cell(x)}, with independent random signs σ.
- Let p₀ ≡ ½.

*Claim.*
- The total-variation distance between the n-sample law under the σ-mixture and under p₀ is at most P(two samples share a cell), which is at most n²/(2M).
- So any certificate B with P(B ≤ R*) ≥ 1 − α for *every* law satisfies, under p₀ (floor ½), P(B ≤ ½ − δ) ≥ 1 − α − n²/(2M).
- Letting δ → ½ and M → ∞, an everywhere-valid certificate is forced to 0 on the constant law.

*Proof.* Within each cell the mixture labels are iid Bern(½) as long as no two samples share a cell. Couple the two laws on that event and apply the validity of B under each p_σ. ∎

This agrees with Antos, Devroye & Györfi (1999).

**Remedy: a test-retest substudy.** A random, outcome-blind subset of m rows gets a second, conditionally independent label. Then:

- **(a) Assumption-free floor.** E 1{y ≠ y′} = g_i *exactly*. A KL lower bound plus Hoeffding–Serfling transport to the n-row floor gives a valid certificate with no shape assumption (`substudy_bound`).
- **(b) Falsification of A(γ).**
  - The extrapolation runs on the rows *outside* the substudy, and the zero-lag measurement on the substudy, so the two are independent.
  - T = extrap(rest) − Q_rep has E T ≤ 0 under A(γ).
  - The test uses Hoeffding with a 2-colouring, plus a random-split term.
- **(c) Pipeline.** Use the largest γ ≤ γ_cap that is not falsified, at α/2, and take the maximum with the substudy bound at α/2.

**Honest limit.** Falsification power grows with m. A micro-scale overshoot of size η needs m ≳ log(1/α)/η². The pipeline is valid in adversarial designs only to the extent that the test has power there. The substudy bound alone is always valid.

## 2. Weak point 2: choosing γ (theory)

**Proposition 2 (mixtures of power laws).** Suppose that on the lag window of the levels used, D(s) = Σ_k c_k s^{γ_k} with c_k ≥ 0 and every γ_k ≥ γ. More generally, suppose D(s)/s^γ is nondecreasing. Then A(γ) holds.

*Proof.* For s_a < s_b and each k, c_k s_a^{γ_k} = c_k s_b^{γ_k}(s_a/s_b)^{γ_k} ≤ (s_a/s_b)^γ · c_k s_b^{γ_k}. Sum over k. ∎

**Local regimes**
- p in C² with bounded density: D = ½E|∇p · Δx|² + O(s^{3/2}) ∝ s, so γ = 1. The first-order term gives equality, so γ = 1 sits *exactly on the edge*.
- Jumps of p across a (d−1)-rectifiable boundary: the crossing probability is ∝ |Δx|, so D ∋ c·s^{1/2}.
- Smooth plus jumps: D = c₁s + c₂s^{1/2}, so **γ = ½ is provably safe**, while γ = 1 can fail.

**The remaining assumption** splits into three parts:
1. No unresolved micro-variation (only replicates can check this).
2. The coarsest lag stays below the variogram range, i.e. no saturation.
3. The small-lag exponent is at least γ.

**Increments test.** A(γ) implies Δ_bc ≥ ((s_c/s_b)^γ − 1)·Δ_ab for level triples. The test uses Janson's inequality with χ\* = 3, because each matching is one colour class. *Empirically it had no power at n = 4000.* It is kept only as a diagnostic.

## 3. Weak point 3: validity resting on slack

There are two slack-free checks.

1. **Brier floor as the target.** Ḡ = mean(1 − |p_i|²) is exactly the Bayes risk of the Brier score. PYTHIA-X certifies it with *no Jensen step*. Validity then rests only on A(γ) and concentration, which makes it a cleaner test of the assumption. It is also a proper-scoring-rule floor, and it doubles as the information measure (Gini/Brier information).
2. **Expectation-level check from the true laws.** Compute (E Q_a − t·E Q_b)/(1 − t) − Ḡ, with no sampling and no Jensen. A positive value means A(γ) failed *in expectation*.

## 4. Weak point 4: squared loss

**Fixed-bet capital bounds** (`betting_lower` / `betting_upper`).
- For any *fixed* λ ∈ [0, 1/m), E Π_i(1 + λ(z_i − m)) = Π_i(1 + λ(μ_i − m)) ≤ exp(λ Σ_i(μ_i − m)) ≤ 1 whenever the *average* mean is ≤ m.
- Markov's inequality then gives a valid test for independent, non-identically distributed [0,1] variables. (Predictable bets would need a common mean; fixed bets do not.)
- A geometric λ-grid is Bonferroni-charged.
- The bound adapts to the realized variance and skewness without estimating either.

The lower side needs only z ≥ 0, so it is **range-free**. The declared range enters only the upper bound of the far level, weighted by t.

**Deep levels.** Seven dyadic levels shrink t to 0.01–0.03, which nearly removes that range dependence. The cost is that A(γ) must hold over a wider lag window: the saturation risk in item 2 of the remaining assumption.
