# Agreement-Field Information (AFI): an information theory for text that rides on a table

Code: `pythia/core/agreement_field.py`.
Experiments: `research/peras_v5/afi_20261017/`.
Status: development. A systematic literature check is needed before any priority claim.

## 0. Motivation

Shannon information needs distributions, so for text it needs a language model. Model-based information ("V-usable information", ΔAUC, ΔBrier) changes when the model changes.

For records made of a **structured part X** (a table) and a **free-text part T**, we want something different:
- a notion of information about an outcome Y that is defined *without* modelling text;
- that is *measured* by observable events;
- that *composes* across the table and the text;
- and that comes with finite-sample guarantees.

## 1. Primitive: agreement, not probability of a single text

Let p(r) = P(Y | r) be the outcome law of record r. The single primitive is the **agreement kernel**

  κ(r, r′) = P(Y = Y′) = ⟨p(r), p(r′)⟩,

the probability that two records' outcomes coincide, a cross-collision probability (Rényi order 2). Everything is built from two observable events: two outcomes **agree** or **disagree**.

| Shannon notion | AFI notion | Formula |
|---|---|---|
| Entropy H(Y) | Disagreement entropy H_A(Y) | 1 − Σ_k P(Y = k)²: two independent draws disagree |
| Conditional entropy H(Y \| R) | H_A(Y \| R) | E[1 − \|p(R)\|²]: two draws *sharing the same R* disagree. This is the Brier Bayes floor |
| Mutual information | Agreement information I_A(Y; R) | H_A(Y) − H_A(Y \| R) = E\|p(R) − p̄\|², the variance of the posterior |
| Chain rule | **Exact** chain rule | I_A(Y; X, T) = I_A(Y; X) + I_A(Y; T \| X), by the law of total variance |
| Data processing | Holds | I_A(Y; f(R)) ≤ I_A(Y; R), by Jensen |
| Fano-type link to error | Two-sided | f_K(H_A) ≤ R₀₁ ≤ H_A |

AFI is target-directed, not symmetric: it measures information *about Y*. Its algebra is the ANOVA, or Sobol, decomposition of the Bayes posterior.

## 2. Measurement: relational pair designs

A *design* selects pairs of rows using only relations between records, never outcomes:
- same table cell;
- text-near at scale s;
- both;
- neither.

**Design identity.** If rows are independent given records and every row is used equally,

  E[disagreement | design] = H_A(Y | R) + D(design),

with D = mean ½|p_i − p_j|² ≥ 0 over the selected pairs. (Cell-replicate designs carry an extra O(1/m) term.)

**Information is a contrast between designs:** how much more often outcomes agree when records are related.

| Quantity | Design contrast | Certificate (no model, no smoothness) |
|---|---|---|
| I_A(Y; X), table information | random pairs − same-cell pairs | **two-sided**: cells are exact replicates |
| H_A(Y \| X), table floor | same-cell pairs | **two-sided** |
| I_A(Y; T \| X), text information beyond the table | same-cell − (same-cell ∧ text-near) | **lower bound** |
| H_A(Y \| X, T), joint floor | same-cell ∧ text-near | **upper bound** |
| I_A(Y; T), text information | random − text-near | **lower bound** (iid rows) |
| Resolution spectrum I_A(Y; T \| X; s) | contrast at text scale s | lower bound per scale: *where* in the text the signal lives |
| Design co-information | I(X) + I(T) − [I(X) + I(T \| X)] | exploratory; > 0 means echo/redundancy, < 0 means synergy |
| Test of "text adds nothing beyond the table" | H₀: I_A(Y; T \| X) = 0 | exact finite-sample p-value |

**Asymmetry principle.**
- Structured fields have replicates, so their information is measurable **from both sides**.
- Free text has no replicates, so its information is measurable **from below only**, and its floor **from above only**, unless an assumption is added (PYTHIA-X γ) or replicates exist (duplicate notes or retests).
- This is the AFI form of the no-free-lunch result (Antos, Devroye & Györfi 1999) and of the nugget problem.

**Geometry freedom.** Validity holds for *any* outcome-blind text geometry, including one learned from outcomes on *other* rows (cross-fitting). The geometry affects only power. That makes the choice of geometry a clean, separate problem: **validity comes from the design and power from the geometry.**

## 3. The new task: Value-of-Text Certification (VTC)

**Input.** n records of (structured row, free-text note, outcome).

**Output: a certified report**, all statements simultaneous at level α, with no model and no smoothness constant:
1. Table information I_A(Y; X) ∈ [L, U].
2. Table floor H_A(Y | X) ∈ [L, U].
3. Text-beyond-table information I_A(Y; T | X) ≥ L, with a p-value for "the note adds nothing".
4. Joint floor H_A(Y | X, T) ≤ U.
5. Text-only information I_A(Y; T) ≥ L.
6. The resolution spectrum, and the sign of the design co-information (echo vs synergy).

**Variant: extraction-completeness certification.** When X is a structured extraction *from* the notes, statement 3 certifies how much outcome information the extraction pipeline still leaves in the free text.

**Decision use.** Is it worth building NLP on top of the structured data for this outcome? Does a proposed extraction schema capture the outcome-relevant content?

## 4. Closest prior work

- **Rényi quadratic entropy and quadratic mutual information** (information-theoretic learning, Príncipe). Uses kernel or Parzen estimates on continuous spaces; estimation, not certification.
- **Sobol / ANOVA sensitivity indices.** The same variance algebra, applied to model outputs.
- **Chatterjee's ξ and the Azadkia–Chatterjee conditional dependence coefficient** (Annals of Statistics 2021). These use nearest-neighbour graphs in X and in (X, Z) to *estimate* conditional dependence for real-valued Y. They are the closest relatives of the design contrast for I(T | X). They are consistent estimators, but they are not finite-sample certificates and have no table/text asymmetry.
- **Multimodal PID** (Liang et al. 2023) and **V-usable information** (Xu et al. 2020; Ethayarajh et al. 2022). Model-relative.

**Claimed new (subject to a fuller review)**
- Treating agreement as the primitive for text information.
- The design calculus with exact contrasts.
- The asymmetry principle (two-sided for tables, one-sided for text).
- Finite-sample, model-free certificates for text information beyond a table.
- The resolution spectrum.
- The VTC task and its extraction-completeness variant.
