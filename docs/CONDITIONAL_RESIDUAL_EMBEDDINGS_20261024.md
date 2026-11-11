# Conditional residual embeddings: notes explain table residual variance; certifying it; turning it into risk (2026-10-24)

- **Data:** testbed v2. That is real MTSamples notes (3,423) with a planted surgical duration driven by seven kinds of note content; the true mean is known for every case.
- **Split:** fixed and outcome-free: FIT 1,331 / CAL 1,023 / TEST 1,069. FIT trains every projection, PCA, ridge and stack weight, and also chooses the tuning (taming quantile, outcome cap). CAL certifies. TEST audits. One risk-forecast scale estimate in `cre_eval2.py` is currently computed from in-sample FIT residuals, not OOF residuals; this is a known optimistic-estimation issue and should be replaced before treating the risk method as stable.
- **Status:** exploratory development on a known-truth testbed. It is not Mayo evidence.

**Code**
- `research/instrumented_20261019/`:
  - `embed_vm.py`: LiteLLM `text-embedding-3-small`, note-only and table-joint variants;
  - `cre_eval.py`: variance explained, gate, forecasts, labeler;
  - `cre_eval2.py`: risk computed from the residual forecast;
  - `cre_bound_final.py`: certified duration information with FIT-only tuning.
- `pythia/core/instrumented.py`: `eb_scalar_bound`, `tune_on_fit`, `tame_scores`.
- Tests: `tests/test_score_learning_and_agent.py`, 4 passed.

## What the literature says (and what is new here)

- **Fusing notes with structured data helps, but modestly and unevenly.** Benchmark gains for mortality and length of stay are often not significant. Gains concentrate in specific phenotypes. Shortcut and keyword reliance is a recurring risk. [Benchmark 2026](https://arxiv.org/html/2602.23614v1); [multimodal routing](https://arxiv.org/html/2607.09982).
- **Joint serialisation (table as text + note) is a strong simple baseline.** It beats gated and concatenation fusion in recent LLM studies, though the gain over gradient boosting is small. [LLMs as unified multimodal learners](https://arxiv.org/html/2607.15380v1).
- **Table-context conditioning of embeddings** (e.g. CASE: tabular-LM embeddings with an in-context data prefix, PCA-reduced) helps most in low-data regimes, and slightly hurts numeric-heavy tasks. [CASE 2026](https://arxiv.org/html/2608.03565).
- **Contrastive EHR–note alignment** exists. [2025](https://arxiv.org/pdf/2505.17643)
- **LLM-driven feature engineering for EHR** exists: FeatEHR-LLM, MedFeat, FeatLLM. [FeatEHR-LLM](https://arxiv.org/pdf/2604.22534); [FeatLLM](https://arxiv.org/html/2404.09491v2)
- **Partialling text out of covariates** is used in causal ML. [Text DML](https://arxiv.org/html/2601.01511v1)

**Important correction to the novelty framing:** Floodgate already targets the same population estimand for squared loss: the mMSE gap between conditioning on the table and conditioning on table plus text. It provides asymptotic lower confidence bounds using a working regression function, with a model-X version and a doubly-robust extension. Therefore this project must not claim that prior work does not certify incremental outcome variance or conditional predictive value. The plausible distinction to test is narrower: a bounded finite-sample projection certificate for a predeclared text-reader score, without requiring a known text-given-table law, extended to patient clusters and strict point-in-time availability. Whether that distinction is mathematically valid and novel remains unproven; Floodgate and other conditional-variable-importance methods must be implemented on the same splits and estimand. The per-case risk labeler is a separate contribution and needs its own calibration/coverage evidence.

## Methods

1. **Note embedding:** `text-embedding-3-small` of the note, reduced to 64 dimensions by PCA fitted on FIT.
2. **Joint table+text embedding:** the same model applied to a serialised table context ("specialty = …; table-based predicted duration = … min") followed by the note. This is one embedding of the case as the table sees it plus what the note adds.
3. **Conditional residual embedding (CRE, table-conditioned):** [E64, E16 × table-cell one-hot]. It lets the residual direction depend on the table context, shrunk by a ridge penalty.
4. **Residual learning:** every reader's features are centred within table cells. A ridge regression then targets the within-cell residual r = y − f(table), which partials out the table, as in double ML. Readers are combined by FIT out-of-fold stacking.
5. **Comparison readers:** the LLM schema reader (`gpt-6-luna`), OpenMed disease and pharma entity bags, TF-IDF, and an oracle of the planted concepts (the reference ceiling).

## Results

### A. How much of the within-cell duration residual variance the note explains (TEST R²)

The true explainable share is **0.366**.

| Reader | TEST R² | Share of the explainable |
|---|---|---|
| **Stack of all real readers** | **0.169** | 46% |
| LLM schema reader | 0.118 | 32% |
| Joint table+text embedding | 0.058 | 16% |
| OpenMed disease | 0.054 | 15% |
| OpenMed pharma | 0.042 | 11% |
| Note-only embedding | 0.033 | 9% |
| TF-IDF | −0.013 | — |
| CRE with table × embedding interactions | −0.302 (over-fits at n = 1,331) | — |
| Oracle concepts | 0.393 | ≈ 100% |

**The notes do explain a substantial part of the variance the table leaves:** about half of what is explainable, once readers are stacked. Each reader type carries different pieces (anesthesia and blood loss; comorbidities; drugs), so stacking beats every single reader.

**Conditioning helps when done cheaply.** Prefixing the table context to the note before embedding (joint) nearly doubles the embedding's value over the note alone (0.058 vs 0.033). Explicit table × embedding interactions over-fit at this size.

### B. Certifying it (Pythia)

The certified quantity is the within-cell variance of E[capped residual | note], in min². It uses the empirical-Bernstein projection bound, with the taming quantile and cap chosen on FIT only, at α/9.

| Reader | Certified (min²) | Truth (min²) | Share of truth |
|---|---|---|---|
| **Stack of all real readers** | **13.7** | 253 | 0.05 |
| TF-IDF | 7.0 | 253 | 0.03 |
| LLM schema reader | 3.9 | 253 | 0.02 |
| Joint embedding | 3.2 | 253 | 0.01 |
| Oracle (reference) | 92.5 | 253 | 0.37 |

- **Bound validity:** the bound never exceeded the truth.
- **The headline:** a positive **conditional lower bound for the within-cell variance of the capped residual conditional mean captured by the declared reader scores**. This is not a universal measure of all information in arbitrary notes, nor a proof that the method improves on Floodgate. The estimate is far below the plug-in estimate, which recovers about 118 min², or 44%. The gap is the concentration inequality: the range term of empirical Bernstein is governed by max |h| × cap.
- **Next for the bound:**
  - betting or Bentkus-type inequalities with heterogeneous ranges;
  - larger n: Mayo has about 13,000 cases, about 4× this testbed.

**Earlier attempts that certified 0:**
- The binary W30-miss target certified 0 for real readers. That target throws away most of the information.
- The first scalar version certified 0, because shifting the outcome inflated the empirical variance.
- Both are fixed by bounding h·r directly.

### C. From the bound to the risk labeler

The risk forecast is p̂ = P(|r̂ + e| > 30), with e ~ N(0, s²). Here r̂ is the stacked residual forecast. This converts a residual forecast into per-case risk rather than directly learning the binary miss outcome. However, `cre_eval2.py` currently estimates `s` from in-sample FIT residuals (`rr[fit] - rhat[fit]`), despite its header saying OOF. That scale is likely optimistic because the same FIT rows trained the residual model; the TEST risk scores are still held-out evaluations of this algorithm, but the one-seed 15% improvement should be considered provisional until `s` is estimated from nested OOF predictions or a dedicated calibration split and the result is repeated across seeds.

| Risk forecast (TEST) | MAE vs true risk | Brier |
|---|---|---|
| Metis bins (cell miss rate) | 0.0849 | 0.1901 |
| Residual model without text (cell mean + Gaussian) | 0.0796 | 0.1875 |
| **+ stacked readers** | **0.0719** | 0.1834 |
| + LLM schema reader | 0.0752 | 0.1834 |
| + OpenMed pharma | 0.0793 | 0.1878 |
| + joint embedding | 0.0850 | 0.1894 |
| + oracle (reference) | 0.0267 | 0.1754 |

- Per-case risk error drops by **15%** relative to Metis bins.
- The certified labeler (forecast bins, m = 250 declared in advance) still certifies directions only for the oracle forecaster: 68% of CAL cases, 0 wrong on TEST truth. That is up from 33% with the binary approach.
- Real-reader forecasts are not yet sharp enough to certify group directions with about 1,000 calibration cases.

## Spend (LiteLLM)

- **Embeddings:** about 2.9 M tokens, about USD 0.06.
- **No new chat calls.**
- **Running total on the key this project:** about USD 1.1 (estimates from list prices).

## Next

1. **Bound:** variance-adaptive betting / Bentkus inequality on h·r with heterogeneous ranges, and duration-scale (not binary) certificates as the default Pythia text certificate.
2. **Risk labeler:** compute risk from the certified residual direction (section C) as the Metis default, with the forecast-bin certificate. Plan m from the target gap before outcomes.
3. **Representation:**
   - joint table+text embedding as a default reader;
   - an LLM *residual reader* that is given the table context and asked what in the note moves this case away from the table's expectation;
   - local OpenMed hidden-state embeddings for Mayo, where notes cannot leave the machine.
4. **Mayo:** only after the provenance fixes (conservative note cutoff, enforced lineage, nested FIT/CAL/TEST). Use local embeddings and readers only, unless de-identified LiteLLM use is approved.
