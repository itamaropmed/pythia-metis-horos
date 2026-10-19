# Table-anchored text information: a definition of "signal in text" measured against the tabular record

Code: `pythia/core/anchored_information.py`.
Suite: `research/peras_v5/anchored_20261017/`.

## 1. What already exists (checked 2026-10-17)

| Line of work | What it defines | Limits for our purpose |
|---|---|---|
| Shannon / perplexity / entropy of language | Information *in* text about itself | Says nothing about a target Y |
| **V-usable information** (Xu et al., ICLR 2020); **pointwise V-information**, dataset difficulty (Ethayarajh, Choi & Swayamdipta, ICML 2022) | Information *usable by a model family V* | Relative to a chosen model family and needs training; an estimate, not a certificate; changes when V changes |
| **Partial information decomposition (PID) for multimodal data** (Williams & Beer 2010; Liang et al., NeurIPS 2023) | Redundancy, uniqueness and synergy between modalities (e.g. text and tabular) | Estimators rely on fitted models or optimization over distributions; no finite-sample guarantee; Shannon-PID definitions are themselves debated |
| **Added value of clinical notes** (e.g. Dutch GP notes, PMC10654855; systematic review medRxiv 2022.01.17.22269400; multimodal EHR, EMNLP 2021) | ΔAUC or ΔBrier of a text+structured model versus a structured-only model | Model-dependent; can be negative from overfitting, or miss interactions the model class cannot represent |

**Gap.** None of these gives a *model-free, finite-sample, certified* amount of target-relevant information that text carries **beyond the structured record**. None of them uses the tabular record as the *yardstick* for text.

## 2. The idea: tabular data as the coordinate system for text

In tabular data we know how to find signal, because rows with the same record are **exact replicates of X**. Pythia turns replicates into assumption-free certificates.

Text has no exact replicates, but text arrives *attached to* a tabular record. So:

- **The table defines the frame.** Cells are patients with identical structured records, e.g. the same age band, sex and lab flag.
- **The text's signal is whatever varies *within* a cell.** Two patients with the same record but different notes either share an outcome law or they do not. Any systematic difference in their outcome laws that the note can tell apart is information the text adds beyond the table.

## 3. Definitions (Brier geometry; p_i = P(Y | X_i, T_i))

| Name | Definition | Meaning |
|---|---|---|
| Table floor Ḡ_X | Σ_c w_c (1 − \|p̄_c\|²) | Best achievable Brier score using the table only |
| Joint floor Ḡ_XT | mean_i (1 − \|p_i\|²) | Best using table and text |
| **Anchored text information ΔI(T \| X)** | Ḡ_X − Ḡ_XT = Σ_c w_c Var_c(p) | Outcome-law variance *inside* table cells that the text resolves: what the note adds beyond the chart |
| Table information I(X) | (1 − \|p̄\|²) − Ḡ_X | Signal in the structured record |
| Text-equivalent table value | The smallest set of table columns S with Ḡ_{X_S} ≤ Ḡ_T | "This note is worth age + stage + grade": an exchange rate between text and columns |
| Brier PID | unique_T = ΔI(T\|X), unique_X = ΔI(X\|T), shared = I(X) − unique_X | A variance-type (Brier) decomposition. Unlike Shannon PID it is a sum of squares, so the pieces are certifiable |

**Why Brier and not Shannon.** The Brier floor and its pair-disagreement identity, E 1{Y_i ≠ Y_j} = (g_i + g_j)/2 + ½|p_i − p_j|², make every quantity estimable from *label agreement of paired rows*. There is no density estimation, no model and no embedding training.

## 4. Theorem (assumption-free certificate of anchored text information)

**Assumptions.** Rows are independent given (X, T); the table X is discrete, or a declared discretization; the text geometry and within-cell pairing are outcome-blind.

Inside each table cell c (m_c rows):
- Q_all is the within-cell all-pairs disagreement, a U-statistic: E Q_all = Ḡ_XT + Σ_c w_c (m_c/(m_c − 1)) V_c.
- Q_txt is the disagreement of a perfect matching built inside each cell by text similarity: E Q_txt = Ḡ_XT + D_txt, with D_txt ≥ 0.

Hence

  ΔI(T | X) ≥ ((m\* − 1)/m\*) · (Q_all − Q_txt − dev),

where:
- m\* is the smallest cell size used;
- dev is a Hoeffding term in which the two dependent statistics combine as (√a + √b)².

**Also assumption-free.** A two-sided interval for the table floor Ḡ_X, from Q_all, and an upper bound on the joint floor Ḡ_XT, from Q_txt.

**Interpretation.** The certified text information is the amount by which *notes that read alike* agree on the outcome more than *random patients with the same chart* do. No text model, no embedding training, no smoothness constant and no γ is needed.

**Model-guided pairing.** Pairs may be formed by any outcome-blind score, for example a within-cell kNN label score cross-fitted on the other fold. This tightens D_txt without affecting validity. Folds are certified separately at α/2. Within-fold variance is at most total within-cell variance, so the sum stays a valid lower bound.

**Caveats**
1. If the table is a coarsening of continuous fields, ΔI is measured *relative to the coarsened table*, which overstates what the text adds to the full-resolution table.
2. Power scales as 1/√n. Small corpora certify only large anchored information.
3. Positive dependence between paired rows, such as copied notes with copied outcomes, would bias Q_txt downward and must be audited.
