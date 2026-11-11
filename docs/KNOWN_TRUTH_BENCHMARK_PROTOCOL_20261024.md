# Known-truth benchmark package v1.0: audit and a fair evaluation protocol (2026-10-24)

**Package:** `pythia_five_known_truth_benchmarks` (seed 20261024). It holds five benchmarks of 6,000 rows each: train 3,600 / calibration 1,200 / test 1,200, with eras 2019–21 / 2022–23 / 2024–25.

**Audit script:** `research/known_truth_benchmarks_20261024/audit_package.py`. Output is in `audit_v1.jsonl`.

**What the audit touched:**
- observed files, for label-free checks;
- oracle files on the **train split only**, for the shortcut checks.

No calibration or test outcomes were used. No reader has been built yet.

## 1. Audit findings

| Check | semantic_frailty | contextual_anticoag | temporal_deterioration | negation_uncertainty | distributed_composition |
|---|---|---|---|---|---|
| Notes before prediction time | 100% | 100% | 100% | 100% | 100% |
| Null is a within-cell permutation | yes | yes | yes | yes | yes |
| True V / within-cell residual variance | 0.34 | 0.25 | 0.41 | 0.31 | 0.37 |
| Text-signal prevalence | 25% | 13% | 41% | 34% | 14% |
| Distinct note strings (of 6,000) | 940 | 958 | 2,025 | 954 | 5,285 |
| **Test notes seen verbatim in train** | **96%** | **98%** | **82%** | **96%** | 14% |
| **Bag-of-words AUC for the latent (train 5-fold)** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| Effect structure | +18, all cells | +22, ablation only | +18, all cells | +16, all cells | +24, all cells |

### Blocking issues for *semantic* claims

1. **Oracle leak in the observed file.** `table_reference_prediction` equals the oracle's `conditional_mean_table_only` to within 5·10⁻⁶ in every benchmark.
   - It must be excluded from every reader and model input.
   - The table model must be fitted on train.
   - An "oracle table" arm may be reported, but only labelled as such.
2. **The signal is lexically separable.**
   - A bag of unigrams recovers the binary latent perfectly in all five benchmarks.
   - In temporal_deterioration a single word decides it: "worsened" vs "remained stable", |corr| = 1.0.
3. **Template reuse across splits.** 82–98% of test notes appear verbatim in train in four of the five benchmarks, so a lookup table scores perfectly.
4. **Simple effect structure.** The latent is binary with a constant shift. Only the anticoagulation benchmark depends on context.

**Consequence.** v1.0 is a good **pipeline-validity** benchmark:
- certificate coverage;
- the null false-claim rate;
- labeler correctness;
- power versus n.

It **cannot** separate semantic readers from keyword readers.

### Proposed fixes for v1.1

- Drop `table_reference_prediction`, or move it to the oracle file.
- **Template-disjoint splits:** paraphrase families used in test never appear in train or calibration. Also hold out one paraphrase *generator*, such as a different LLM family.
- A **graded or continuous latent**, and partial evidence:
  - implicit mentions;
  - hedges with intermediate probability;
  - distractor sentences about other people or the past ("mother had…", "resolved in 2019");
  - contradictions that need recency ordering.
- **Target lexical separability:** a train bag-of-words AUC ≤ 0.80 while the oracle reader stays near 1.0. Report this gap in the manifest.
- **Context dependence in at least three benchmarks:** effect varying by cell, by age or complexity, or by the time between note and prediction.
- **20 replicate datasets per benchmark** (different seeds, same design). A single dataset gives only one coverage outcome per method; validity needs replicates.
- A `true_risk_w30` column, P(|Y − f| > 30 | X, T) under the generator's noise law, or an explicit statement that the noise is Gaussian.

## 2. Evaluation protocol

### 2.0 Freeze before any calibration or test access

Write a preregistration file, hashed, containing:
- the readers and their prompts;
- the portfolio size J;
- α = 0.05;
- the outcome cap (declared in minutes);
- labeler m;
- the metrics;
- the null count K.

Record the data file hashes. Development uses **train only**.

### 2.1 Input firewall

| Allowed reader/model inputs | Forbidden |
|---|---|
| `entity_id` (grouping only), timestamps, `cell`, `age`, `scheduled_minutes`, `procedure_complexity`, `comorbidity_count`, `table_noise_feature`, `note_text_signal` | `table_reference_prediction`, every oracle column, `note_text_null_permuted_within_cell` (null runs only), `split` as a feature, any calibration/test outcome |

**Enforcement:**
- A loader returns only the allow-listed columns.
- The oracle file is opened only by `score.py`, *after* predictions and certificates are written to disk with their hashes.

### 2.2 Pipeline (identical for every method)

1. **Table model, fitted on train.** Use CatBoost, plus ridge with cell × spline terms; the better by train CV becomes f.
   - Train residuals are 5-fold out-of-fold.
   - Calibration and test residuals come from the train-fitted f.
2. **Readers and text models** are trained on train only, with hyperparameters chosen by train CV (entity-grouped, era-blocked).
3. **Certification on calibration**, with the entity as the unit:
   - each certificate at α/J;
   - J is declared and includes every reader or agent proposal ever scored.
4. **Audit on test.**

### 2.3 Questions and metrics

| Question | Metric | Truth source |
|---|---|---|
| Does the text add information beyond the table? | Lower confidence bound on V = within-cell Var(E[r \| X, T]); validity = P(bound > V_true); tightness = bound / V_true | oracle `true_incremental_conditional_mean` (capped version if a cap is declared) |
| Null behaviour | False-claim rate over K = 200 fresh within-cell permutations of the calibration text, with a Clopper–Pearson CI; must be ≤ α | permutation |
| Duration prediction | Test MAE and R² vs table only; gap to the oracle-optimal MAE | oracle means, `true_optimal_mae_risk` |
| Per-case risk | Test MAE / Pearson / Spearman vs the true W30 risk; Brier; calibration slope | `true_risk_w30` (requested) |
| Labeler | Certified share on calibration; wrong-direction rate on test truth | oracle |
| Robustness | All of the above on (a) template-disjoint test, (b) held-out-generator test, (c) era shift | v1.1 |
| Power | Certified share vs calibration n ∈ {300, 600, 1,200}, by subsampling entities | — |

### 2.4 How nulls are run (finite-sample, repeated)

- **Cheap readers** (TF-IDF, embeddings, rules) re-run the *full* calibration certification for each permutation.
- **LLM readers** are label-free functions of a single note. So we permute their cached per-note outputs within cell; that is equivalent to permuting the text and costs no API calls.
- The training side is never permuted for a calibration-null claim. A separate "train-null" run permutes train text too, to check that the whole pipeline (agent included) does not manufacture signal.
- **Report:** the false-claim rate with a 95% CI, plus the bound distribution. A single bound is not required to be exactly 0.
- **Coverage under the alternative** comes from the 20 replicate datasets: P(bound > V_true) ≤ α, with a CI.

## 3. Baseline suite

**A. Table only.** CatBoost / XGBoost, TabPFN-v2, RealMLP or TabM.

**B. Table + text predictors.** For each, report test MAE/R² and risk metrics. Each is also turned into a score h for the certificate, so every method is certified the same way.
1. TF-IDF (1–3-grams) + ridge on the residual.
2. Bag-of-words logistic on the latent. This is the *lexical-shortcut* reference and must be reported.
3. Local sentence embeddings (MiniLM / BGE-small) + ridge.
4. API embeddings (text-embedding-3-small; note-only and joint table+note) + ridge.
5. Fine-tuned Bio_ClinicalBERT / ModernBERT regression on the residual, train only.
6. CatBoost with native text features.
7. AutoGluon-Multimodal (tabular + text).
8. TabSTAR (tabular foundation model with text).
9. LLM serialization, zero-shot and few-shot (table + note → minutes).
10. LLM feature engineering (FeatLLM-style).
11. **Ours:**
    - the table-conditioned residual reader;
    - the symbolic sample space;
    - the Metis concept agent (train-only decisions);
    - OpenMed entities;
    - the stack of all of these.

**C. Inference rivals for "does the text add information".**
1. Williamson et al. nonparametric variable importance: R² gain with a CI, cross-fitted.
2. Split LOCO with inference (Lei et al.).
3. Double-ML incremental R² with an entity bootstrap.
4. The GCM conditional-independence test (a test only, no magnitude).
5. Our certificates: betting (default), Maurer–Pontil (old), and CLT (reference only).

**D. Risk-labeler rivals.**
1. Metis bins.
2. Direct classifiers on 0/1 miss.
3. Mondrian-by-cell split conformal and CQR intervals turned into P(miss).
4. Isotonic / Venn–Abers calibrated classifiers.
5. Ours: the residual forecast (Gaussian and empirical errors) and the certified labeler.

### Fairness rules

- Same train data and splits for everyone.
- Same hyperparameter-search budget (e.g. 20 configurations by train CV).
- Same LLM spend cap per method.
- Prompts are frozen before any calibration/test access.
- The **regex/bag-of-words reference is always shown**, so a "semantic" claim means beating it on template-disjoint test data.

**Disclosure.** I (Claude) have read this package's README, which names the five signal types, and the train-split audit above, including the top separating unigrams. To keep our own readers from being tuned to v1.0:
- our readers are frozen as they are now;
- semantic claims are made only on a v1.1 regenerated with paraphrase families and a generator I have not seen.

## 4. Cost

| Item | Estimate |
|---|---|
| LLM residual reader on the distinct notes (~10k across five benchmarks, cached) | about USD 2.5 at the current model price |
| Embeddings | about USD 0.2 |
| Null runs on cached outputs | USD 0 |
| Replicates (×20) | local readers by default; LLM readers on 3 replicates unless more budget is allowed |
