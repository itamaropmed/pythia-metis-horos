# Preregistration: blind evaluation of Pythia/Metis on benchmark kit v2.1 (frozen 2026-10-24T12:03Z)

**What was frozen, and when:**
- Frozen **before any v2.1 file was opened.**
- The runner and scorer were smoke-tested only on the earlier v1.0 package, which is a development package and not part of this evaluation.

**Scope of the claims:**
- Behaviour of certificates and forecasters on generated known-truth data.
- **No SOTA claim and no clinical-validity claim.**

## Frozen code (sha256)

| File | sha256 |
|---|---|
| `research/blind_v2_20261024/pythia_blind_eval.py` | `99551ba7b33d6e654f590126ef1baea09ea6fe9c82b0ee1a5d1ea88399630368` |
| `research/blind_v2_20261024/score_with_oracle.py` | `dab8f6a57f7b64a8eb283aa29053c532c22f01a64da5de467c673796438b04f1` |
| `pythia/core/tight_bounds.py` | `d343536361506d38ea2da5d9c54a76027dfb3b274498642fb7c5a5a04bccb0db` |
| `pythia/core/instrumented.py` | `7af7aded494386c94dfa1f460ae9645fd6b912f6e6af39167d33cf9c7618efa0` |
| `metis/reader_gain.py` | `046f0e3b398aa7c1130177f18896714cac9ae81fe8fae3fd39d372201a5af13f` |

**Environment:** numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5. Seed 20261024.

**Change control.** Any change after this point is a versioned amendment, with the reason and its timestamp recorded *before* outputs on the 19 withheld replicates exist. Analyses added after outputs exist are labelled **exploratory**.

## Data handling

- The runner reads one **observed-only** file per (design, replicate). It never opens oracle files.
- **Allow-listed inputs:**
  - `row_id`, `entity_id` (grouping only), `split`, `cell`;
  - numeric table columns;
  - the note text;
  - timestamps, used only for audit.
- **Excluded automatically:** columns matching `oracle | true_ | latent | conditional_mean | noise_sd | reference_prediction | risk | seed | null`, plus the outcome and miss columns.
- The resolved column lists are written to `run_meta.json` for inspection.
- **Development uses train rows only.** Calibration outcomes are used only by the declared certificates and the labeler. Test outcomes are never used by the runner.
- **Order of operations:**
  1. Replicate 00 is used to check that the pipeline runs on the v2.1 schema.
  2. Outputs for all 20 replicates are produced with the frozen code.
  3. Output hashes are sent to the data owner.
  4. Only then does the data owner run `score_with_oracle.py`.

## Pipeline (identical for every design and replicate)

1. **Table model (train only).** Two candidates:
   - ridge with cell one-hot, standardized features, hinge splines and cell × feature interactions;
   - HistGradientBoosting.

   The candidate is chosen by 5-fold entity-grouped train CV MSE. Train residuals are out-of-fold.
2. **Outcome cap** = 3 × train out-of-fold residual SD, rounded up to 5 minutes. This defines the certified target, V of the capped residual.
3. **Text readers.** All are trained on train to predict the within-cell residual, with penalties chosen by train CV. J = 6, declared:

   | Reader | Description |
   |---|---|
   | `bow_unigram` | Lexical-shortcut reference |
   | `tfidf_word12` | Word 1–2-gram TF-IDF ridge |
   | `tfidf_char35` | Character 3–5-gram TF-IDF ridge |
   | `tfidf_word12_x_cell` | Table-conditioned: TF-IDF × cell interactions |
   | `tfidf_svd_hgb` | 100-dimension SVD + cell → gradient boosting (nonlinear) |
   | `STACK` | Train out-of-fold least-squares stack of the five above |

   **No LLM, embedding-API or transformer readers in this preregistration.** The budget does not allow them across 100 replicate files. Any such arm will be a separate, labelled amendment.
4. **Certificates on calibration.**
   - Each patient or entity is one independent unit.
   - α = 0.05 / J per reader.
   - Taming quantile and bet size are tuned on train out-of-fold scores only.

   | Certificate | Status |
   |---|---|
   | **Predictable-plug-in betting** | **Primary** |
   | Maurer–Pontil empirical Bernstein | Secondary |
   | CLT | Reference; not a finite-sample certificate |
   | Williamson-style influence-function R² gain (lower bound × residual variance) | Secondary rival; asymptotic |
   | Percentile bootstrap of the plug-in | Reference |
5. **Nulls.**
   - K = 200 within-cell permutations of the calibration scores. This is equivalent to permuting calibration notes for these per-note, train-fitted readers.
   - The false-claim rate is the share of permutations with bound > 0, for each method.
   - Also, one full-pipeline run (train + calibration) on the provided null text column for `tfidf_word12`.
6. **Predictions, for scoring by the data owner.**
   - **Duration:** f, and f + cell offset + h for each reader.
   - **W30 risk forecasters:**
     - Metis bins;
     - direct logistic (table + TF-IDF);
     - direct HistGradientBoosting (table);
     - Gaussian residual forecast without text;
     - Gaussian residual forecast with STACK;
     - empirical (per-cell, train out-of-fold errors) residual forecast with STACK.
   - The miss reference is identified on train by an exact match of the provided miss column. If no match is found, this is reported and f is used.
7. **Labeler.**
   - Forecast tertiles (edges from train) × cell.
   - Local-risk v2 certificate on calibration with m = 250 entities, declared.
   - Per-row directions are exported.

## Primary endpoints (per design, pooled over 20 replicates)

**E1 – Certificate validity.** P(betting bound > V_true) ≤ α/J, with a Clopper–Pearson CI. With 20 replicates × 6 readers this only detects gross failures.

**E2 – Null false-claim rate.** Over 20 × 200 permutations, the rate is ≤ α/J, with a CI.

**E3 – Tightness.** Mean of bound / V_true for the betting certificate vs Maurer–Pontil vs the influence-function R² rival. Coverage is reported alongside, because a rival that is tighter but under-covers is not better.

**E4 – Semantic vs lexical.** Certified V and test MAE-to-true-mean for each reader vs `bow_unigram`. On this kit (train unigram AUC ≤ 0.635 by design), a reader is called "beyond lexical" only if its certified bound exceeds `bow_unigram`'s in at least 16 of 20 replicates.

**E5 – Risk.** Test MAE, Pearson and Spearman of each W30 forecaster vs the exact Gaussian risk.

**E6 – Labeler.** Share of rows with a certified direction, and the wrong-direction rate on test truth.

## Expectations stated in advance

These local lexical readers are expected to certify **little** on designs whose signal is semantic, compositional or recency-based. Low certified values there are an expected and informative outcome, not a failure of validity. Validity (E1, E2) is expected to hold for betting, Maurer–Pontil and the provided-null run. CLT and the bootstrap may over-claim.
