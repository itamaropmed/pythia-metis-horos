# Agreement-Field Information for natural-language text: definitions, theorems, proofs

Companion to `AGREEMENT_FIELD_INFORMATION_20261017.md`.
Code:
- `pythia/core/agreement_field.py`
- `pythia/core/dynamic_geometry.py`

## 1. Setting

**Records and outcomes.** A record is r = (x, t):
- x ∈ 𝒳 is a *finite* table value (a cell);
- t ∈ 𝒯 is a text: a finite string over any alphabet, with no metric, no vector space and no language model assumed.

The outcome is Y ∈ {1, …, K}. The data (X_i, T_i, Y_i), i = 1..n, are independent. All statements condition on the records unless "iid" is stated.

**Laws.** p_i = P(Y_i = · | X_i, T_i) ∈ Δ_K. Write g(p) = 1 − |p|², the Gini or collision complement.

**Table cells.** C_x = {i : X_i = x}, with m_x = |C_x| and w_x = m_x/n. The within-cell mean law is p̄_x = mean_{i ∈ C_x} p_i.

**Text geometry.** Any map φ: 𝒯 → (M, d) into a metric space: a bag of n-grams, an e5 or BERT embedding, a diffusion map, or a learned projection. φ is a *tool*, not part of the definitions.

## 2. Definitions (agreement-field quantities, realized-record form)

| Name | Definition |
|---|---|
| Agreement kernel | κ_ij = ⟨p_i, p_j⟩ = P(Y_i = Y_j) |
| Joint disagreement entropy (Brier floor of table + text) | H_A(Y \| X, T) = (1/n) Σ_i g(p_i) |
| Table disagreement entropy | H_A(Y \| X) = Σ_x w_x g(p̄_x) |
| **Text information beyond the table** | I_A(Y; T \| X) = H_A(Y \| X) − H_A(Y \| X, T) = Σ_x w_x V_x, with V_x = mean_{i ∈ C_x} \|p_i − p̄_x\|² |
| Table information | I_A(Y; X) = g(p̄) − H_A(Y \| X) |
| Text information (iid form) | I_A(Y; T) = E\|P(Y \| T) − P(Y)\|² |

The algebra is exact:
- **chain rule** I_A(Y; X, T) = I_A(Y; X) + I_A(Y; T | X), by total variance;
- **non-negativity**;
- **data processing** under coarsening;
- **Fano-type sandwich** f_K(H_A) ≤ R₀₁ ≤ H_A.

## 3. Theorems

### Theorem 1 (pair identity)

For i ≠ j, P(Y_i ≠ Y_j) = (g(p_i) + g(p_j))/2 + ½|p_i − p_j|².

**Proof.** By independence, P(Y_i ≠ Y_j) = 1 − ⟨p_i, p_j⟩. Also |p_i − p_j|² = |p_i|² + |p_j|² − 2⟨p_i, p_j⟩. ∎

### Theorem 2 (within-cell identities)

(a) Let Q_x be the all-pairs disagreement U-statistic in cell x. Then E Q_x = g(p̄_x) + V_x/(m_x − 1).

(b) Let M_x be a perfect matching of C_x that is σ(X, T)-measurable, i.e. it may use the texts but not the outcomes. Let Q_x^φ be its disagreement rate. Then

  E Q_x^φ = H_x + D_x^φ,

where H_x = mean_{i ∈ C_x} g(p_i) and D_x^φ = mean_{(i,j) ∈ M_x} ½|p_i − p_j|² ≥ 0.

**Proof.**
- (a) Average Theorem 1 over all pairs in the cell. Then use mean_{i≠j} ½|p_i − p_j|² = (m/(m − 1)) V_x and g(p̄) = H_x + V_x.
- (b) Average Theorem 1 over the matching. Each row appears exactly once, so the g-terms average to H_x. ∎

### Theorem 3 (certified text information beyond the table: assumption-free)

Let Q = Σ_x w_x Q_x and Q^φ = Σ_x w_x Q_x^φ, and let m\* = min_x m_x. With probability at least 1 − α,

  I_A(Y; T | X) ≥ ((m\* − 1)/m\*) · [ Q − Q^φ − √( log(1/α) · (√a + √b)² / 2 ) ],

with a = Σ_x w_x² / ⌊m_x/2⌋ and b = 1/N_φ, where N_φ is the number of matched pairs.

The bound holds **for every text geometry φ**. It requires no smoothness, no language model and no embedding quality. The geometry affects only how large Q − Q^φ is.

**Proof.**
1. By Theorem 2: E(Q − Q^φ) = Σ_x w_x [V_x · m_x/(m_x − 1) − D_x^φ] ≤ (m\*/(m\* − 1)) · I_A(Y; T | X).
2. Concentration of Q. Each Q_x is a U-statistic of order 2 with a kernel in [0, 1]. Hoeffding's (1963) argument writes it as an average over permutations of means of ⌊m_x/2⌋ independent terms. Its centred moment generating function is therefore sub-Gaussian with proxy 1/(4⌊m_x/2⌋). Cells are independent, so Q has proxy a/4.
3. Concentration of Q^φ. It is a mean of N_φ independent indicators (disjoint pairs), so its proxy is b/4.
4. Combining. The two statistics share rows. By Hölder's inequality, the sub-Gaussian proxy of a sum of dependent sub-Gaussians is at most (√a + √b)²/4.
5. Apply Chernoff and rearrange. ∎

### Theorem 4 (asymmetry: text information has no assumption-free upper bound)

Suppose the law of T given X is atomless, so texts never repeat. Then for every n, every δ < ½ and every procedure U_n with P(U_n ≥ I_A(Y; T | X)) ≥ 1 − α for *all* laws, there is a law with I_A(Y; T | X) = 0 under which P(U_n ≥ δ²·c) ≥ 1 − α − ε, where c > 0 depends only on K and ε → 0.

In words: a valid upper bound on text information must stay non-trivial even when the text carries nothing. By contrast, table information has a two-sided certificate (Theorem 2a with between-cell contrasts).

**Proof (sketch).**
1. Partition 𝒯 into M atoms of equal probability.
2. Under law A, set p(x, t) = p̄_x + δ·σ_{atom(t)}·e, with independent random signs σ and a fixed direction e, so I_A ≈ δ². Under law B, p ≡ p̄_x, so I_A = 0.
3. If no two samples share an atom, the n-sample laws of A (mixed over σ) and B coincide.
4. The probability that two samples share an atom is at most n²/(2M). Let M → ∞, and apply the validity of U_n under A. ∎

### Theorem 5 (dynamic and learned geometries remain valid)

Split the rows at random into ordered batches 1..B. Let ℱ_{b−1} be generated by all records together with the outcomes of batches < b. Suppose the batch-b geometry φ_b (and its matching) is ℱ_{b−1}-measurable, i.e. it is *predictable*.

Let Z_b be the within-batch version of Q − Q^φ, and let I_b be the within-batch text-beyond-table information. Then:

(i) E[Z_b | ℱ_{b−1}] ≤ (m\*/(m\* − 1)) · I_b.

(ii) With probability at least 1 − α,

  Σ_b w_b I_b ≥ ((m\* − 1)/m\*) · [ Σ_b w_b Z_b − √( log(1/α) · Σ_b w_b² π_b / 2 ) ],

with predictable proxies π_b.

(iii) Σ_b w_b I_b ≤ I_A(Y; T | X).

**Proof.**
- (i) Batch-b outcomes are independent of ℱ_{b−1} given the records, so Theorem 2 applies conditionally.
- (ii) The increments are conditionally sub-Gaussian with ℱ_{b−1}-measurable proxies. Apply Azuma–Hoeffding to the martingale differences.
- (iii) Replacing a cell mean by within-batch cell means can only lower the within-cell variance. Dropped rows contribute ≥ 0 to the right-hand side. ∎

**Consequences**
- The geometry can be *learned and updated as data arrive*, for example as posterior-probability coordinates, NCA projections or fine-tuned embeddings, with no penalty in validity.
- The guarantee is anytime-valid in the batch index.
- Two-fold cross-fitting is the special case B = 2 with a union bound.

### Proposition 6 (optimal geometry and the agreement objective)

For a fixed matching structure, D^φ = 0 if and only if matched rows have equal laws. Hence the *information-optimal text geometry is the posterior map* t ↦ p(· | x, t). Its learnable surrogate is to minimise matched-pair disagreement, E 1{Y_i ≠ Y_j} over φ-neighbours, on *past* data.

This is precisely the Neighbourhood Components Analysis objective (Goldberger et al. 2004) and the posterior-geometry learner in `dynamic_geometry.py`. The certificate therefore defines its own representation-learning loss.

### Proposition 7 (resolution limit)

Let ρ_φ = Σ_x w_x D_x^φ / I_A(Y; T | X) be the fraction of information the geometry fails to resolve. Certification succeeds with high probability once

  (1 − ρ_φ) · I_A(Y; T | X) ≳ 2·√( log(1/α) / n ),

up to the factor m\*/(m\* − 1). This is a sample-complexity law for certifying text signal.

### Proposition 8 (diffusion-time geometry)

For a diffusion map Ψ_t of a kNN graph on φ(T), the family {Ψ_t}_t is outcome-blind. A Bonferroni portfolio over diffusion times t is therefore valid by Theorem 3, and t is the dynamic resolution dial: small t is local and lexical, large t follows the data manifold.

## 4. Status

- Theorems 1–3 and 5 are proved above, at the level of detail given.
- Theorem 4 is given as a sketch.
- Propositions 6–8 are statements with short arguments.

**Empirical checks**
- `research/peras_v5/afi_20261017/`: ten generated datasets and MTSamples.
- `research/peras_v5/dynamic_20261017/`: dynamic geometries and e5 embeddings.

**Closest prior work**
- Azadkia & Chatterjee (2021), nearest-neighbour conditional dependence: estimation only.
- Neighbourhood Components Analysis (Goldberger et al. 2004).
- Coifman & Lafon (2006), diffusion maps.
- Information-theoretic learning and Rényi-2 quadratic mutual information (Príncipe).
