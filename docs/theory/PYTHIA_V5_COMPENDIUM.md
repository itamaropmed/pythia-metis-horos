# Pythia v5 compendium: definitions and results in one place

This note gives condensed statements only. Proofs, discussion and literature are in the source notes; each item names its source.

**Notation**
- Records (X_i, T_i, Y_i), i = 1..n, independent; Y ∈ {1..K}.
- p_i = P(Y_i = · | X_i, T_i); g(p) = 1 − |p|² (Gini / collision complement); Ḡ = mean_i g(p_i).
- Table cells C_x, with sizes m_x and weights w_x = m_x/n.
- f_K(G) = ((K − 1)/K)(1 − √(1 − K G/(K − 1))), the Cover–Hart inversion.
- Loss conventions: "Brier units" means 1 − |p|², summed over classes. Brier/2 is the usual binary Brier score.

---

## A. Identities: the measurement layer

**A1 Pair identity** (AFI_TEXT_THEOREMS, Thm 1). P(Y_i ≠ Y_j) = (g_i + g_j)/2 + ½|p_i − p_j|².

**A2 Equal-floor identity** (MODULUS_FREE_EXTRAPOLATION). For every outcome-blind perfect matching M,

  E Q(M) = Ḡ + D(M),  D(M) = mean_{(i,j)∈M} ½|p_i − p_j|² ≥ 0.

The floor term is the same for every perfect matching. In variogram language (PYTHIA_X_STRENGTHENING), D is the semivariogram of the label law and Ḡ is the nugget.

**A3 Within-cell identity** (AFI Thm 2; PYTHIA_U Lemma 1). The within-cell all-pairs disagreement has conditional mean exactly

  (cell floor) + (m/(m − 1)) × (within-cell variance of p).

**A4 Squared loss.** ½(Y_i − Y_j)² has mean σ̄² + ½(μ_i − μ_j)². The same structure gives MSE floors.

## B. Floors and ceilings without models

**B1 Ceiling U1** (MODEL_FREE_SANDWICH). Ḡ ≤ min_ℓ Q(M_ℓ)_U, and R₀₁ ≤ Ḡ.
- No assumption is needed.
- Any outcome-blind geometry may be used: text TF-IDF, embeddings, or the predicted probabilities of a model trained on other rows (U3).

**B2 Information lower bound I1** (MODEL_FREE_SANDWICH). (1/n) Σ |p_i − p̄|² ≥ ((n − 1)/n)(Q_all,L − Ḡ_U).

**B3 PYTHIA-X floor under A(γ)** (MODULUS_FREE_EXTRAPOLATION).
- Assumption A(γ): D(a) ≤ (s_a/s_b)^γ D(b) for nested scales a < b.
- Conclusion: Ḡ ≥ (Q(a)_L − t Q(b)_U)/(1 − t), with t = (s_a/s_b)^γ, and R₀₁ ≥ f_K(Ḡ).

**B4 Prop. 1, no assumption-free floor without replicates** (PYTHIA_X_STRENGTHENING). Some assumption, or some replicates, is necessary. A 20% retest substudy gives an assumption-free anchor.

**B5 Prop. 2, choice of γ** (PYTHIA_X_STRENGTHENING). If D(s)/s^γ is nondecreasing on the lag window, A(γ) holds. γ = ½ is safe for smooth-plus-jump laws; γ = 1 can fail at class boundaries.

**B6 Replicate certificates** (REPLICATE_PAIR_CERTIFICATES). Exact replicates, including orbits of a declared nuisance group of canonicalisations, give assumption-light 0-1 floors through unbiased minorants.

**B7 Table side for any loss** (DOMINATED_SAA, DEPENDENCY_GRAPH_DOMINATION, SMOOTH_DOMINATION).
- Fixed-action domination gives one-sided, range-based concentration of the SAA floor.
- It extends to dependency-graph data with a realized-period estimand, and to continuous X through second-order drift penalties.

## C. Information carried by text relative to a table

**C1 AFI** (AGREEMENT_FIELD_INFORMATION). Text information beyond the table is the agreement-kernel within-cell variance

  ΔI(T | X) = H_A(Y | X) − H_A(Y | X, T),

measured by contrasting cell-pair and text-neighbour-pair disagreement.

**C2 AFI Theorem 3.** Assumption-free lower bound on ΔI for every text geometry.

**C3 AFI Theorem 4 (asymmetry).** No non-trivial assumption-free upper bound on text information exists. Table information is two-sided; text information is lower-only.

**C4 AFI Theorem 5 (predictable geometry).** Geometries learned on earlier batches keep validity (martingale argument, Azuma). Learning the representation is free.

**C5 Prop. 6 (optimal geometry).** The information-optimal text geometry is the posterior map t ↦ p(· | x, t). Its surrogate objective is matched-pair disagreement, which is the NCA objective.

**C6 Prop. 7 (resolution limit).** Certification needs (1 − ρ_φ) I ≳ 2√(log(1/α)/n).

## D. Natural Language Information (NLI)

Source: NATURAL_LANGUAGE_INFORMATION_NLI.

**D1 Definition.** nli_ℓ(x, t) = d_ℓ(p(x, t), p̄(x)), the Bregman divergence of the proper loss ℓ. NLI_ℓ = E nli_ℓ(X, T).

**D2 Operational theorem.** NLI_ℓ = R\*_ℓ(X) − R\*_ℓ(X, T) for every proper loss.
- Log loss gives I(Y; T | X).
- Brier gives E|p − p̄|², the canonical NLI.

**D3 Axioms A1–A7**
- non-negativity;
- faithfulness (zero iff conditional independence);
- echo invariance;
- chain rule;
- data processing (the extraction-completeness gap);
- re-encoding invariance;
- stratum decomposition.

**D4 Transfers.** I(Y; T | X) ≥ ½ NLI_quad. For binary outcomes the 0-1 value is at most √(NLI_quad/2). The 0-1 value can be 0 while NLI > 0.

**D5 Uniqueness.** The quadratic NLI is the only Bregman NLI that is a linear functional of the pair-agreement kernel, i.e. the only one observable from single labels.

**D6 Variational form.**

  NLI_quad = sup_{h: E[h|X]=0} (E⟨e_Y − p̄, h⟩)² / E|h|²,

attained at h = p − p̄.

**D7 Finite-sample certificate.**
- Construction: fit any score on fold A; on fold B, centre it leave-one-out within cells; then NLI ≥ 2 kl⁻((Ĉ − dev)₊²/H).
- Validity: it holds at the null boundary.
- Rate: it becomes positive once NLI ≳ 1/m, against m^{-1/2} for the pair route.
- Cohort form: the same bound without the population lift.

**D8 Exact test.** Within-cell permutation of Σ yᵢhᵢ. It is essentially equivalent to the local permutation test (Kim et al. 2022): NLI's contribution is the *magnitude*, not detection power.

**D9 Lower-only.** As in C3.

## E. LP layer and the unified certificate

**E1 LP-1** (LP_RELAXATION_AND_TEXT_GEOMETRY). Partial identification over the variables (G, V, D_ℓ, F_ℓ):
- certified boxes;
- exact constraint D ≤ 2V;
- A(γ) relaxed by ε.

It yields the floor L(γ, ε), the tipping slack ε\*, and falsification of A(γ) when the program is infeasible.

**E2 LP-2.** Moment LP for the 0-1 floor from certified moments: the Hausdorff problem.

**E3 PYTHIA-U** (PYTHIA_U_UNIFIED_CERTIFICATE).
- The single estimand is conditional on the design (cohort S), and every component is a bound for independent, non-identical labels.
- The LP combines pair boxes, the table box (McDiarmid), the cross-fitted NLI bound (V ≥ Σ w_f nli_f), the held-out model ceiling and A(γ).
- Outputs, all simultaneous:
  - joint floor [L, U];
  - table floor;
  - table information;
  - NLI;
  - falsification and ε\*;
  - loss transfers;
  - population lift.
- Table-conditional variant: per-stratum certificates with α split.

**E4 PYTHIA-G3** (`research/peras_v5/g3_20261017/G3_RESULTS_20261017.md`).
- Any-loss neighbour-label leave-one-out ceiling.
- Fixed-action table floor.
- Three-gap report.
- Per-sample irreducibility score and Learn-then-Test certified-easy groups.

## F. Open audit items

1. **U-statistic concentration under non-identical rows.** Several notes (MODEL_FREE_SANDWICH I1, AFI Thm 3, LP-1's table box) invoke Hoeffding's U-statistic bound conditional on the records. Hoeffding's argument is stated for iid samples.
   - When the conditional means differ across permutations, the averaged moment-generating-function step needs an extra argument.
   - The statements are valid for population targets with iid rows.
   - For conditional (cohort) targets, PYTHIA-U uses McDiarmid instead (constant √2 worse), which is valid unconditionally.
   - **To do:** either prove the conditional version, or switch the remaining routes to McDiarmid or to population targets.
2. **Joint floor.** The lower side remains assumption-tagged by necessity (B4, C3). Its tightness at γ = ½ is weak on real text.
3. **Table box.** A variance-adaptive bound is needed; McDiarmid gives ±0.04 at n = 8000.
