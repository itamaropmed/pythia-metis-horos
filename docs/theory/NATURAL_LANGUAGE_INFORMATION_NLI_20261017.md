# Natural Language Information (NLI): the expected newly introduced information of a text, given the table

- **Code:** `research/peras_v5/nli_20261017/methods.py`. The projection certificate and test are in `nli_projection`; the pair and geometry route is in `nli_certificate`.
- **Experiments:** `research/peras_v5/nli_20261017/` (results in `NLI_RESULTS_20261017.md`).
- **Naming.** In NLP, "NLI" usually means natural language *inference*. Here NLI means **Natural Language Information**: how much a note tells you about the outcome *beyond what the structured record already says*. When both meanings could appear together, write *NL-Information*.

---

## 0. Setting and notation

**Data.** Each record is (X, T, Y):
- X is the structured record (table row), taking values in a finite or countable set of *cells*.
- T is the free text of the note.
- Y ∈ {1..K} is the outcome, with one-hot code e_Y.

**Conditional laws**
- The *reader's law* is p(x, t) = P(Y = · | X = x, T = t) ∈ Δ_K.
- The *table law* is p̄(x) = P(Y = · | X = x) = E[p(X, T) | X = x].

**Losses.** A loss ℓ(y, a) has generalized entropy H_ℓ(q) = inf_a E_{Y∼q} ℓ(Y, a), which is concave. The loss is *proper* if a = q attains the infimum, and *strictly proper* if q is the only minimiser. For concave, differentiable H, the Bregman gap is

  d_ℓ(p, q) = H_ℓ(q) + ⟨∇H_ℓ(q), p − q⟩ − H_ℓ(p) ≥ 0.

**Bayes risks.** R\*_ℓ(X) = inf_f E ℓ(Y, f(X)) and R\*_ℓ(X, T) = inf_f E ℓ(Y, f(X, T)).

## 1. Definition

**Definition 1 (pointwise and expected NLI).** For a proper loss ℓ, the *natural-language information of note t for patient x* is

  nli_ℓ(x, t) = d_ℓ( p(x, t), p̄(x) ),

the divergence between what the reader believes after reading the note and what the table alone says. The **Natural Language Information** of T about Y given X is its expectation:

  **NLI_ℓ(T → Y | X) = E[ d_ℓ( p(X, T), p̄(X) ) ].**

**Canonical (quadratic) NLI.** Brier loss has H(q) = 1 − |q|², which gives

  NLI ≡ NLI_quad = E |p(X, T) − p̄(X)|²   (Brier units; binary case: 2 E(p₁ − p̄₁)²).

The *resolution* of the text is NLI / E[1 − |p̄(X)|²]. It is the share of the table's irreducible disagreement that the text resolves.

**Non-differentiable losses.** For 0-1 or pinball loss, define NLI_ℓ through Theorem 1 as the Bayes-risk reduction R\*_ℓ(X) − R\*_ℓ(X, T).

## 2. What it means: the operational theorem

**Theorem 1 (operational equality).** For every proper loss ℓ with finite risks,

  NLI_ℓ(T → Y | X) = R\*_ℓ(X) − R\*_ℓ(X, T).

*Proof.*
1. By properness, R\*_ℓ(X) = E H(p̄(X)) and R\*_ℓ(X, T) = E H(p(X, T)).
2. The linear term vanishes: E[⟨∇H(p̄(X)), p(X, T) − p̄(X)⟩ | X] = 0, because E[p | X] = p̄.
3. Taking expectations of d_ℓ therefore leaves E[H(p̄) − H(p)]. ∎

**Instances**

| Loss | NLI_ℓ |
|---|---|
| Log loss | I(Y; T \| X), the conditional mutual information |
| Brier | E\|p − p̄\|², the expected squared innovation of the posterior |
| 0-1 | the reduction in the best achievable error rate |

So NLI is the *value of reading the note* for **every** decision-maker at once. This is the DeGroot (1962) value of information, specialised to text over a table. For the uninformed reader, the table is the prior.

**Corollary (information budget).** For every proper ℓ,

  R\*_ℓ(∅) − R\*_ℓ(X, T) = [R\*_ℓ(∅) − R\*_ℓ(X)] + NLI_ℓ(T → Y | X).

This is the split *table information + natural-language information*. It is the exact version of the *data gap* used in PYTHIA-G3.

## 3. Axioms and properties

Write NLI for NLI_ℓ with ℓ strictly proper unless stated.

- **(A1) Non-negativity.** NLI ≥ 0.
- **(A2) Faithfulness.** NLI = 0 iff Y ⊥ T | X. Strict properness makes d_ℓ(p, q) = 0 iff p = q.
- **(A3) Echo invariance.** If T = g(X, U) with U ⊥ (X, Y), then NLI = 0. This holds however much of the table the note repeats, in any wording or style. The unconditional I(Y; T) *does* count the echo. NLI counts only what is new. The decomposition I_ℓ(Y; T) − NLI_ℓ(T | X) = I_ℓ(Y; X) − I_ℓ(Y; X | T) is the co-information: it is positive for echo-dominated notes and negative for synergy.
- **(A4) Chain rule.** NLI(T₁, T₂ | X) = NLI(T₁ | X) + NLI(T₂ | X, T₁). This telescopes Theorem 1. It applies to note sections, multiple notes, or time-ordered notes.
- **(A5) Data processing and extraction completeness.** For any representation φ(T) (embedding, extracted concepts, LLM summary), NLI(φ(T) | X) ≤ NLI(T | X). Equality holds iff φ is sufficient: p(X, T) = p(X, φ(T)) a.s. The gap NLI(T | X) − NLI(φ(T) | X) = E d_ℓ(p(X, T), p(X, φ(T))) is the **information an extraction pipeline throws away**.
- **(A6) Re-encoding invariance.** NLI is unchanged under any bijection of T, such as tokenisation, lossless paraphrase or translation. NLI is a property of the law, not of the string.
- **(A7) Stratum decomposition.** NLI = Σ_x P(X = x) NLI_x, where NLI_x = E[d_ℓ(p(x, T), p̄(x)) | X = x]. This gives the *map* of where in the table the notes carry signal.

## 4. Loss transfers

**Proposition 2.**

1. **Log loss.** I(Y; T | X) ≥ ½ NLI_quad (in nats). *Proof:* Pinsker gives KL ≥ ½|p − p̄|₁², and ℓ₁ ≥ ℓ₂.
2. **0-1 loss, binary.** V₀₁ = R\*₀₁(X) − R\*₀₁(X, T) ≤ E|p₁ − p̄₁| ≤ √(NLI_quad / 2). For K classes, V₀₁ ≤ E|p − p̄|_∞ ≤ √NLI_quad.
3. **0-1 loss can miss information.** V₀₁ = 0 is possible while NLI_quad > 0: the note shifts risk without flipping the best decision. NLI_quad is the *decision-agnostic* summary. The 0-1 value is not.

## 5. Why the quadratic NLI is canonical

From single labels, the observable second-order object is pair agreement: for independent rows i ≠ j, P(Y_i = Y_j | x_i, t_i, x_j, t_j) = ⟨p_i, p_j⟩.

**Theorem 3 (uniqueness).** Suppose a Bregman NLI_ℓ is, for every law, a linear functional of the agreement kernel κ(x, t; x′, t′) = ⟨p(x, t), p(x′, t′)⟩ together with the marginal law of (X, T). Suppose also that H_ℓ is continuous. Then H_ℓ(q) = c + ⟨b, q⟩ − a|q|² with a ≥ 0. In other words, ℓ is an affine transform of the Brier loss, and NLI_ℓ = a · NLI_quad.

*Sketch.*
1. **Quadratic H gives a linear functional.** If H is quadratic, NLI = a(E⟨p, p⟩ − E⟨p̄, p̄⟩). The first term is κ on the diagonal. The second is κ averaged over two independent texts in the same cell. Both are linear in κ.
2. **Conversely, reduce to a functional equation.** Take X constant and T two-valued, with p ∈ {u, v} and weights λ, 1 − λ. The Jensen gap J(λ; u, v) = H(λu + (1 − λ)v) − λH(u) − (1 − λ)H(v) must then be a linear function of ⟨u, u⟩, ⟨u, v⟩, ⟨v, v⟩ whose coefficients depend only on λ.
3. **Solve it.** Matching across λ and using continuity forces J = λ(1 − λ)·a|u − v|². By the Jordan–von Neumann characterisation, H minus a|q|² is then affine. ∎

So the quadratic NLI is the only NLI that pairs of single-label records can see directly. Fisher–Rao or log-loss NLI would need higher-order agreement among replicates. This matches the identifiability remark in the LP/geometry note.

## 6. Variational representation (the key tool)

**Theorem 4 (variational NLI).** Let 𝓗 be the set of square-integrable vector fields h(X, T) ∈ ℝ^K with E[h | X] = 0 and E|h|² > 0. Then

  **NLI_quad = sup_{h ∈ 𝓗} ( E⟨e_Y − p̄(X), h(X, T)⟩ )² / E|h|²**,

and the supremum is attained by the *innovation* h\* = p − p̄. In the binary case,

  NLI_quad = 2 sup_h (E[Y h])² / E h²  over scalar h with E[h | X] = 0.

*Proof.*
1. Remove the label noise: E⟨e_Y, h⟩ = E⟨p, h⟩, since E[e_Y | X, T] = p.
2. Remove p̄: E⟨p̄, h⟩ = E⟨p̄, E[h | X]⟩ = 0.
3. So the numerator is E⟨p − p̄, h⟩. Cauchy–Schwarz gives (E⟨p − p̄, h⟩)² ≤ NLI · E|h|², with equality iff h ∝ p − p̄. ∎

**Reading.**
- NLI is the largest squared covariance, per unit energy, between the label's surprise relative to the table and any *text-driven direction that averages to zero within table cells*.
- Any model gives a *lower* bound. A miscalibrated or misspecified model loosens the bound but never invalidates it.
- A perfect model makes the bound tight. The bound is scale-free in h, so the model only has to *rank* patients correctly within cells. Its probabilities need not be calibrated.

**Relation to prior work.** This Cauchy–Schwarz device appears in the *floodgate* method (Zhang & Janson, 2020) for model-free variable importance. Floodgate uses asymptotic inference and needs a model-X conditional law. The numerator is the weighted generalized covariance measure (Shah & Peters 2020; Scheidegger, Hörrmann & Bühlmann 2022). What is new here is the finite-sample, model-X-free version for a discrete table, which follows.

## 7. Finite-sample certificate (assumption-free, valid at the null boundary)

**Split.** Split the rows into folds A and B. On fold A, fit any score q(X, T), for example a table+text classifier. On fold B, with m rows, set

  h_i = q_i − (1/(m_c − 1)) Σ_{j ≠ i, j ∈ c(i)} q_j   (leave-one-out within-cell centering; uses no labels).

**Facts**
- (i) Σ_{i ∈ c} h_i = 0 exactly in every cell.
- (ii) Conditional on the fold-B design 𝒟 = (X_i, T_i)_{i ∈ B} and on fold A, the labels y_i are independent with means p_i, and the h_i are fixed.

**Theorem 5 (NLI certificate).** Let Ĉ = (1/m) Σ y_i h_i, H = (1/m) Σ h_i², and dev = √(H log(1/α₁) / (2m)). Then, with probability at least 1 − α₁ − α₂,

  NLI_quad ≥ 2 · kl⁻_{m, α₂}( (Ĉ − dev)₊² / H ),

where kl⁻ is the Chernoff–KL lower confidence bound for a mean of [0, 1] variables.

*Proof.*
1. **Cauchy–Schwarz in-sample.** By (i), (1/m) Σ p_i h_i = (1/m) Σ (p_i − p̄_{c(i)}) h_i for the *population* cell means p̄_c. Cauchy–Schwarz then gives (1/m) Σ p_i h_i ≤ √(Z̄ H), where Z̄ = (1/m) Σ (p_i − p̄_{c(i)})².
2. **Label noise.** By (ii), Hoeffding with ranges |h_i| gives (1/m) Σ p_i h_i ≥ Ĉ − dev with probability ≥ 1 − α₁. Hence Z̄ ≥ (Ĉ − dev)₊² / H.
3. **Lift to the population.** The Z_i are iid in [0, 1] with mean NLI_quad / 2. The Chernoff–KL lower bound is monotone, so applying it to a lower bound of Z̄ is still valid with probability ≥ 1 − α₂. ∎

**Variants**
- *Two folds with swapped roles:* take the maximum, with a union bound (α/4 per event).
- *In-sample target:* a bound on Z̄ alone, the information the text carries *for this cohort*, avoids the α₂ step.

**Rate (why this matters).** Write ρ = E[(p − p̄)h] / √(E h²) ≤ √(NLI/2). The certificate is

  2(ρ − O(m^{-1/2}))₊².

- It becomes positive once **NLI ≳ 1/m**.
- The model-free pair certificate (AFI: Q_cell − Q_text) has an *additive* slack of order m^{-1/2} on NLI itself. It needs **NLI ≳ m^{-1/2}**.
- The projection route therefore closes a full square-root gap. On the test families at n = 4000, the pair certificate was positive in 0.7% of signal runs and the projection certificate in 61% (`NLI_RESULTS_20261017.md`).

## 8. Exact test of "the note adds nothing"

**Theorem 6 (exact level).** Suppose H₀: Y ⊥ T | X holds and X is discrete. Then, conditional on 𝒟 and fold A, the fold-B labels are exchangeable within each cell. The within-cell permutation p-value of the statistic Σ y_i h_i is therefore exactly valid at every m, for any h fitted on fold A.

*Proof.* This is the standard randomisation argument. The cell partition is fixed, and h depends only on 𝒟 and fold A. ∎

**Remark (equivalence).** The within-cell permutation distribution of Σ y_i h_i equals that of Σ y_i q_i up to per-cell affine maps. So this test is essentially the *local permutation test* (Kim, Neykov, Balakrishnan & Wasserman 2022) applied to a model-improvement statistic. NLI does **not** buy extra detection power over that test. What NLI adds is the **magnitude certificate** of Theorem 5 and its operational meaning (Theorem 1).

## 9. Asymmetry: why there is no upper bound

**Theorem 7 (lower-only).** No procedure gives a nontrivial assumption-free upper confidence bound on NLI. This is restated from the AFI notes, Theorem 4.

**Reason.** A text can carry the outcome through an arbitrarily fine code, for example a hash of a token, which no finite sample distinguishes from noise. *Table* information is two-sided, because cells are exact replicates. *Text* information is certifiable only from below. Upper statements need declared shape assumptions; see PYTHIA-X A(½) and the LP relaxation dial.

## 10. The model-free route (no score model)

AFI pair designs give NLI ≥ Q_cell − Q_text − slack, where Q_text is the disagreement between text-nearest neighbours in a cell.
- Under a static or *predictable* (martingale-valid) geometry, Theorem 5 of `AFI_TEXT_THEOREMS` applies.
- The optimal geometry is the posterior map (Prop. 6 there). The projection of Section 7 is the one-dimensional, *linear* use of that same optimal geometry.
- This explains why it dominates nearest-neighbour pairing.

## 11. Relation to existing notions (honest positioning)

| Notion | Relation to NLI |
|---|---|
| DeGroot information; generalized entropy (Grünwald & Dawid 2004) | NLI_ℓ is the DeGroot value of the note given the table, so the *object* is classical. NLI names and develops it for text over tables: echo invariance, extraction completeness, the stratum map, uniqueness under pair observability, finite-sample certificates. |
| Conditional mutual information | NLI_log. It is not certifiable from below without a loss transfer (Prop. 2). |
| Nonparametric variable importance (Williamson et al. 2021; LOCO, Lei et al. 2018) | NLI_quad is proportional to the population ΔR² importance of T given X. Asymptotic inference for it is invalid at the null boundary. Theorem 5 is finite-sample and boundary-valid. |
| Floodgate (Zhang & Janson 2020) | Same Cauchy–Schwarz idea, but asymptotic and model-X. Here the inference is finite-sample, uses exact replicate cells, and needs no conditional model. |
| V-information and pointwise V-information (Xu et al. 2020; Ethayarajh et al. 2022) | Usable information within a model class. Fixing the class gives "V-NLI" ≤ NLI. The certificate lower-bounds the *unrestricted* NLI. |
| GCM / WGCM, CODEC, local permutation | Tests of Y ⊥ T \| X. They produce p-values, not magnitudes with operational meaning. |
| Partial information decomposition | NLI = unique + synergistic information of T relative to X, in the ℓ-entropy sense (A3). |

**What is new**
1. Defining text information *relative to a table*, with the axioms A1–A7, the extraction-completeness gap and the stratum map.
2. Theorem 3: uniqueness of the quadratic NLI under single-label observability.
3. Theorem 5: a finite-sample, assumption-free, null-boundary-valid lower confidence bound that converts any model into a certified amount of text information. Its rate is 1/m, not m^{-1/2}.
4. Theorem 7: the lower-only asymmetry.
5. Prop. 2: transfers to log and 0-1 decision value.

## 12. The task: NLI certification

**Input:** records (X, T, Y) and a level α.

**Output**
- **(a)** NLI_quad ≥ L, simultaneously with I(Y; T | X) ≥ L/2 nats and the resolution share.
- **(b)** The exact p-value for "the notes add nothing beyond the table".
- **(c)** The per-cell map NLI_x ≥ L_x (Holm).
- **(d)** For any extraction φ, the completeness gap. This needs an upper side on NLI(φ | X), which holds when φ is low-dimensional and discrete, because then φ cells are replicates.
- **(e)** Pointwise nli scores ĥ_i²-type ranks for triage. These are certified only at the group level.
