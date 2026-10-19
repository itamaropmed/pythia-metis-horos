# PYTHIA-U: one simultaneous certificate for the table and its notes

- **Code:** `pythia/core/unified.py` (`unified_certificate`, `stratified_certificate`, `unified_lp`).
- **Tests:** `tests/test_unified.py`.
- **Experiments:** `research/peras_v5/unified_20261018/`.

This note combines every earlier piece into one linear program over a single estimand:
- the table floor from exact replicate cells,
- the text-neighbour pair designs (AFI, LP-1),
- the NLI projection certificate,
- a held-out model,
- the A(γ) shape assumption with its relaxation dial.

All statements hold **simultaneously**.

## 1. One estimand, conditional on the design

**Cohort.** Let S be the rows retained by the text-matching design: every table cell is truncated to a multiple of 2^L rows, chosen at random and outcome-blind. Write p_i = p(X_i, T_i), g_i = 1 − |p_i|², and p̂_c for the mean of p_i over the rows of S in cell c. Write w_c = m_c/|S|.

**Quantities (Brier units)**

| Symbol | Definition | Meaning |
|---|---|---|
| G | (1/\|S\|) Σ_{i∈S} g_i | joint floor: best achievable risk with table + text on these patients |
| V | Σ_c w_c (m_c/(m_c − 1)) mean_{i∈c} \|p_i − p̂_c\|² | **NLI** (in-sample form): text information beyond the table |
| G + V | | table floor: best achievable risk with the table alone, up to the exact m/(m−1) factor |
| G₀ − (G + V) | G₀ is the marginal floor | table information |

**Why everything can be certified together.** Conditional on the design 𝒟 = (X_i, T_i)_{i∈S}, the labels are independent with means p_i. Every certificate below is a concentration bound for independent, non-identical variables given 𝒟. A union bound therefore makes them simultaneous.

**Population version.** The g_i are iid in [0, 1], so a final Chernoff–KL step lifts the interval for G to the population floor at an extra α share.

## 2. The ingredients

| | Certified statement | Why it holds |
|---|---|---|
| (P) pair designs | lo_ℓ ≤ G + D_ℓ ≤ hi_ℓ and lo′_ℓ ≤ G + F_ℓ ≤ hi′_ℓ | Pair identity: P(Y_i ≠ Y_j) = (g_i + g_j)/2 + ½\|p_i − p_j\|². Each level is a perfect matching built from the text alone. |
| (Tb) table | t_L ≤ G + V ≤ t_U | **Lemma 1:** E[Q \| 𝒟] = G + V exactly for the within-cell all-pairs disagreement Q. Changing one label moves Q by at most 2/\|S\|, so McDiarmid applies. |
| (N) NLI | V ≥ Σ_f w_f nli_f | **Lemma 2** below |
| (M) model | G ≤ Σ_f w_f m_U,f | Held-out Brier: E[\|q_i − e_{Y_i}\|² \| 𝒟] = g_i + \|q_i − p_i\|² ≥ g_i |
| (X) exact structure | 0 ≤ D_ℓ ≤ 2V, F_ℓ ≥ 0, V ≥ 0 | ½\|p_i − p_j\|² ≤ \|p_i − c\|² + \|p_j − c\|² for any c, summed over a within-cell matching |
| (A) assumption | D_a ≤ t_ab^γ D_b + ε | A(γ), relaxed by ε. With ε = ∞ the certificate is assumption-free. |

**Proof of Lemma 1.** For a cell with m rows, the all-pairs mean of 1 − ⟨p_i, p_j⟩ over i ≠ j is

  1 − (m²|p̂|² − Σ|p_i|²)/(m(m−1)) = mean g + (m/(m−1)) mean|p_i − p̂|².

Weighting by w_c and summing over cells gives G + V. ∎

**Lemma 2 (cross-fitted NLI).** Split S into halves f = 0, 1. Fit the model for half f on every row *outside* half f, so its predictions q are independent of the half-f labels given 𝒟. Within half f, set h_i = q_i − (leave-one-out cell mean of q). Then Σ_{i ∈ c∩f} h_i = 0 for every cell, so for **any** per-cell constant vector c,

  (1/m_f) Σ_{i∈f} ⟨p_i − c_{c(i)}, h_i⟩ = (1/m_f) Σ_{i∈f} ⟨p_i, h_i⟩.

Cauchy–Schwarz, plus Hoeffding with per-row ranges max_k h_ik − min_k h_ik, gives

  (1/m_f) Σ_{i∈f} |p_i − c_{c(i)}|² ≥ nli_f := (C_f − dev_f)₊² / H_f.

Take c = p̂_c, the cell mean over all of S. Summing over the halves gives (1/|S|) Σ_{i∈S} |p_i − p̂_c|² ≥ Σ_f w_f nli_f, and V is at least this left side. ∎

This is the vector (any-K) form of NLI Theorem 5. It needs no model-X law and no calibration.

## 3. The unified program and its guarantees

**Theorem U1 (simultaneous validity).** Let 𝒞 be the set of (G, V, D, F) that satisfy (P), (Tb), (N), (M) and (X), plus (A) when it is declared. With probability at least 1 − α, the true values lie in 𝒞. Hence:

  **G ∈ [min_𝒞 G, max_𝒞 G],  V ≥ min_𝒞 V,  G + V ∈ [t_L, t_U],  I_X ∈ [m₀L − t_U, m₀U − t_L].**

These hold jointly, together with the loss transfers:
- Brier/2 floor = G/2;
- f_K(G_L) ≤ R\*₀₁ ≤ G_U;
- log-loss floor ≥ G_L;
- I(Y; T | X) ≥ V_L/2 nats.

**α split:** designs α/2, table α/10, marginal α/20, NLI α/10, model α/10, population lift α/10.

**Theorem U2 (sharpness of the combination).** Given the certified boxes, the LP values are the exact identified set for G and V. No other combination of the same certificates gives a valid narrower interval. Each route binds where it is strongest:

| Route | Bound on G | Wins when |
|---|---|---|
| Pair route | G ≤ hi_ℓ | text neighbours are close |
| NLI route | G ≤ t_U − V_L | the text signal is learnable but neighbours are far, as with long noisy notes |
| Model route | G ≤ m_U | the model is close to the posterior |

**Proposition U3 (falsification by NLI).** Under A(γ), the pair boxes give G ≥ L_γ, so V ≤ t_U − L_γ. If the NLI certificate exceeds this, the program is infeasible at ε = 0, and the data **refute A(γ)** at level α.

The smallest ε that restores feasibility, ε_min, measures how strongly the shape assumption is violated. The tipping slack ε\* (where the floor reaches 0) measures how much violation the floor can tolerate. Both are reported.

**Theorem U3 (table-conditional form).** Let s be a coarse table variable, such as note type or age band. Run the certificate inside each stratum at level α_s, with Σ α_s = α. Then Σ_s (|S_s|/Σ|S|) G_s lies in the weighted sum of the stratum intervals, simultaneously with every stratum's own:
- table floor,
- text information V_s,
- binding route.

This is the **conditional map**: how much the notes add *within* each part of the table. The cost is that stratifying spends α across strata and certifies smaller samples. On the global floor it is never tighter than the pooled certificate unless the strata differ strongly.

## 4. What is and is not claimed

**Assumption-free**
- Joint ceiling.
- Table floor (two-sided).
- Table information (two-sided).
- Text information (lower bound).
- Model headroom (lower bound).
- Falsification of A(γ).

**Assumption-tagged**
- The joint lower floor (needs A(γ) with slack ε).
- An upper bound on text information.

Both are impossible without an assumption (AFI Theorem 4, NLI Theorem 7).

**Per-patient labels** (PYTHIA-G3) attach to the same cohort, and the certified-easy groups stay valid by Learn-then-Test.
