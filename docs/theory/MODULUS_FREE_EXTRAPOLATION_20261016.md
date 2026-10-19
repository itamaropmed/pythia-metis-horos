# PYTHIA-X: modulus-free Bayes-floor certificates by certified multiscale extrapolation

Code: `pythia/core/multiscale_pair_extrapolation.py`.
Tests: `tests/test_multiscale_pair_extrapolation.py`.
Suite: `research/peras_v5/novelty_20261016/`.

## The problem it attacks

Every accepted near-replicate or kNN lower bound declares a modulus L (Lipschitz in TV or in the action), and so did every earlier Pythia smooth route. On real geometries such as text embeddings, the realized modulus is astronomically large: about 10⁸–10⁹ on MTSamples TF-IDF/SVD. That happens because the label law and the geometry disagree at tiny scales, and it makes every L-based route vacuous. Rows without exact replicates were the remaining cap in the replicate routes.

## Equal-floor identity

**Setting**
- Labels have laws p_i on K classes.
- M is any outcome-blind *perfect* matching of the rows.
- Q(M) is the fraction of disagreeing pairs.

**Identity**

  E Q(M) = Ḡ + D(M),
  Ḡ = mean_i (1 − |p_i|²),
  D(M) = mean_{(i,j)∈M} |p_i − p_j|²/2 ≥ 0.

This follows from 1 − ⟨p_i, p_j⟩ = (g_i + g_j)/2 + |p_i − p_j|²/2. Because a perfect matching uses every row exactly once, **the floor term Ḡ is the same for every perfect matching**. Only the drift D changes with scale.

The 0-1 floor then satisfies R ≥ f_K(Ḡ) (Cover–Hart inversion plus Jensen). For squared loss, ½(Y_i − Y_j)² gives mean σ² + mean ½(μ_i − μ_j)², with no inversion needed.

## Shape assumption A(γ) replaces the modulus

For nested matchings a < b with mean squared pair distances s_a < s_b:

  D(a) ≤ (s_a/s_b)^γ · D(b).

| γ | Meaning |
|---|---|
| 1 | Drift grows at least like d², the smooth regime (D ≈ c·d² locally) |
| ½ | Drift grows at least like d; laws with jumps or class boundaries |
| → 0 | No assumption; the bound tends to 0 |

γ is scale-free and unit-free. No constant has to be known, and γ acts as a smoothness-order dial.

## Theorem

Let t = (s_a/s_b)^γ < 1. Then

  Ḡ ≥ (Q(a)_L − t·Q(b)_U) / (1 − t)

with probability at least 1 − α_a − α_b. The two bounds are Chernoff–KL bounds: the n/2 pairs inside one matching are independent given X. Then R ≥ f_K(·) of that value.

**Portfolio and construction**
- The portfolio runs over level pairs and γ ∈ {1, ½}, with Bonferroni over candidates.
- Nested matchings are dyadic. At each level, units from the previous level are matched by greedy nearest neighbour on centroids, and their members are paired by a Hungarian assignment. Every level is therefore a perfect matching of the same rows.

**Proof.** By the identity, Ḡ = E Q(a) − D(a) ≥ E Q(a) − t·D(b) = E Q(a) − t(E Q(b) − Ḡ). Rearrange, then apply a union bound for the two one-sided confidence bounds. ∎

## Identification boundary (why some assumption is unavoidable)

Take a law with p(x) ∈ {0.3, 0.7} varying below the sampling resolution (`rowwise_random`, `checker_fine`). Without replicates, it is indistinguishable from p ≡ ½. That fits the no-free-lunch results for Bayes-error estimation (Antos, Devroye & Györfi 1999).

So any certificate that is non-trivial on smooth laws must exclude such laws by assumption. A(γ) does this with a scale-free dial instead of a constant. The suite shows the boundary explicitly: those designs break PYTHIA-X and every estimator, while the oracle-L route stays valid only because it is vacuous.

## Relation to prior work (checked 2026-10-16)

- **Gamma test** (Stefánsson, Končar & Jones 1997; Evans & Jones): regresses ½(ΔY)² on |Δx|² and extrapolates the intercept to estimate noise variance. Estimation only, with no one-sided certification.
- **Okuno & Shimodaira (2020):** extrapolate k-NN estimates of η(x) to k = 0 for classification. Asymptotic rates under Hölder and margin conditions; no finite-sample bounds; not a Bayes-error certificate.
- **Cover–Hart and Fukunaga:** NN-error inversion, asymptotic.

**Claimed new**
1. The equal-floor property of perfect matchings, so drift is the only scale-dependent term.
2. The Brier-drift decomposition for K-class 0-1 loss.
3. One-sided *certified* extrapolation under an explicit, dial-able shape assumption, with finite-sample validity.
4. A demonstration of near-oracle-modulus tightness without a modulus, and of non-vacuous bounds where every modulus route is vacuous.

A systematic review is still needed before any publication claim.
