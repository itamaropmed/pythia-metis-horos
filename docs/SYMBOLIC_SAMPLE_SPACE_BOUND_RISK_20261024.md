# Table-conditioned, embedding-free note information; a tighter bound; the risk labeler; the agentic loop; strict Mayo rerun (2026-10-24)

**Status:** development experiments.
- The public testbed is known-truth, with a planted outcome.
- Mayo results are local-only aggregates under the strict note cutoff.
- Nothing here is preregistered.

**Spend (LiteLLM, list-price estimates):**

| Item | Cost |
|---|---|
| Residual reader, 3,423 public notes | USD 0.81 |
| Agent proposals (4 testbed + 2 Mayo, aggregates only) | about USD 0.003 |
| **Round total** | **about USD 0.82 of the USD 1.00 cap** |

No Mayo note text left the Mac.

## 1. Defining note information without embeddings, conditioned on the table

The object is the conditional information the note carries about the outcome *given the table*:

  V = E | E[r | X, T] − E[r | X] |²,  with r = y − f(X)

Its optimal direction is h\* = E[r | X, T] − E[r | X].

Instead of embedding the note, we make the note's outcome-relevant **sample space** explicit, conditioned on X:

1. **Table-conditioned residual reader.** One LLM call per note, shown the table context: the specialty group, the table's expected minutes and the typical spread. It returns three things, and it never sees outcomes:
   - `shift_min`: a direct estimate of h\*;
   - `p_long30` / `p_short30`: zero-shot tail risks relative to the table;
   - up to 10 **typed, signed atomic propositions** (fact, type, direction, minutes).
2. **Symbolic sample space Ω_T|X.** Each case is (X, A), where A is the finite set of typed atoms.
   - Information is the refinement of σ(X) by σ(X, Q(A)) for a set of questions or concepts Q.
   - By the data-processing inequality it is monotone in refinement.
   - Each refinement is certified by Pythia's portfolio rule.
   - The ledger then decomposes information along the chain of refinements the agent builds.
3. **Concept compiler (Metis agent).** Concepts are regexes proposed by the LLM from the aggregate residual diagnosis and executed locally on all notes.
   - There are no per-note LLM calls after the first pass.
   - So the search over the σ-algebra costs about one chat per round.

Code (`research/instrumented_20261019/`):

| File | Role |
|---|---|
| `residual_reader_vm.py` | the reader |
| `symbolic_reader.py` | features |
| `agent_concepts.py` | diagnose / score / final |
| `llm_json_vm.py` | one LLM call on aggregates |
| `tb_common.py` | shared testbed state |
| `symbolic_eval.py`, `testbed_mae.py` | evaluations |

## 2. Tighter bound: the predictable-plug-in betting certificate

The new code is `pythia/core/tight_bounds.py` (`scalar_cert`, `betting_lower_mean`, `tune_cert`, `crossfit_certificate`).

**Why the old bound was loose.** The Maurer–Pontil term 7·B·log(2/α)/(3n), with B = cap · max|h|, consumed about 90% of the signal.

**The new inequality.** It is the predictable-plug-in empirical-Bernstein supermartingale with a constant bet λ.
- Fan's inequality applied to x_t − μ̂_{t−1} keeps it **valid for non-constant conditional means**, which is exactly the heterogeneous setting here.
- The deviation then scales with the empirical variance; the range enters only through λ < 1.
- λ, the taming quantile, and the declared cap are all chosen on FIT.

**Simulations** (`research/tight_bound_20261024/run_bound_sims.py`): 120 replicates per cell, α = 0.05, cap 60. Each value is the certified share of the true V.

| n | Noise | Maurer–Pontil (old) | **Betting (new)** | 2-fold cross-certified | CLT (reference, not valid) |
|---|---|---|---|---|---|
| 3,400 | Gaussian | 0.070 | **0.122** | 0.108 | 0.218 |
| 3,400 | t3 | 0.082 | **0.130** | 0.116 | 0.234 |
| 8,000 | Gaussian | 0.185 | **0.231** | 0.221 | 0.285 |
| 8,000 | t3 | 0.190 | **0.235** | 0.228 | 0.286 |

- **Validity:**
  - Misses (bound > truth): **0** in every cell and for every method.
  - Null false claims: 0 for Maurer–Pontil, betting and cross-certification.
  - CLT made false claims at **5–10%**, so it is not a certificate.
- **Size:** betting is 1.25–1.75× tighter than Maurer–Pontil.
- **Cross-certification** did not beat a single split at these sizes, because α is split across folds. It stays in the code for larger cohorts.

**Testbed v2, certified within-cell variance of E[r | note]** (min², CAL, cap 60; truth 251.6):

| Reader | MP (old) | **Betting** | Plug-in |
|---|---|---|---|
| Old stack of real readers | 19.6 | 30.8 | 97.9 |
| + symbolic reader | 22.2 | 35.7 | 105.5 |
| **+ agent concepts** | 46.8 | **65.3** | — |
| Oracle | 93.4 | 114.3 | 220.3 |

**Overall: 19.6 → 65.3 min² (3.3×), at α/10 over the declared portfolio. The bound never exceeded the truth.**

## 3. The risk labeler

### 3a. Simulation: which construction tracks the true risk?

The table shows Pearson correlation with the exact true risk on held-out rows, from 40 replicates. Reader noise is 0.5.

| n | Metis bins | Logistic on 0/1 miss | Gradient boosting on 0/1 miss | **Residual forecast** |
|---|---|---|---|---|
| 1,500 | −0.01 | 0.06 | 0.04 | **0.22** |
| 3,400 | 0.00 | 0.09 | 0.07 | **0.30** |
| 8,000 | 0.00 | 0.16 | 0.13 | **0.33** |
| 20,000 | 0.01 | 0.22 | 0.19 | **0.36** |

- The residual forecast is p = P(|r̂ + e| > 30), with the scale taken from FIT out-of-fold errors.
- It also has the lowest MAE at every n.
- **Learning risk from the continuous residual beats learning it from the binary miss.** That makes the residual forecast the Metis default.

### 3b. Testbed v2, held-out TEST vs the true risk

| Forecaster | MAE vs true risk | Pearson | Spearman | Brier | Certified labeler (m = 250) |
|---|---|---|---|---|---|
| Metis bins | 0.0849 | 0.05 | 0.07 | 0.1901 | — |
| Residual, no notes | 0.0796 | −0.11 | −0.10 | 0.1875 | 0% |
| Residual + old readers | 0.0772 | 0.40 | 0.26 | 0.1827 | 0% |
| Residual + symbolic reader | 0.0770 | 0.43 | 0.27 | 0.1823 | 71%, 0 wrong |
| **Residual + agent concepts** | **0.0632** | **0.66** | **0.47** | **0.1791** | **70%, 0 wrong** |
| Residual + oracle | 0.0278 | 0.93 | 0.89 | 0.1752 | 68%, 0 wrong |

**What did not help:**
- a heteroscedastic scale model;
- the LLM's zero-shot tail probabilities, even after recalibration (Pearson 0.08);
- the high-dimensional atom indicators, which over-fit.

## 4. The agentic loop: what the agent found (testbed, FIT-only decisions)

| Round | Accepted concepts (FIT 5×2-fold gain in within-cell R²) | Agreement with the planted concept (diagnostic only) |
|---|---|---|
| 1 | obesity, anticoagulant/antiplatelet, malignancy (+0.087) | 1.00, 0.75, 0.97 |
| 2 | general anesthesia/intubation, heparin, "extensive/complex" (+0.003) | 0.59, 0.69, — |
| 3 | diabetes, hypertension (+0.017) | 0.97, 0.92 |
| 4 | bilateral, multiple procedures, conversion to open (+0.0002) | rejected (irrelevant, correctly) |

**Held-out TEST results:**

| Measure | Before the agent | After the agent | Best possible |
|---|---|---|---|
| Within-cell R² | 0.168 | **0.293** | 0.366 |
| Duration MAE | 20.82 min (table only) | **17.56 min** | 16.13 min |

- R² 0.293 is 80% of what is explainable.
- The MAE gain of **3.26 min** is 70% of the achievable 4.69 min.
- Correlation of the learned direction with the true direction rose from 0.63 to 0.82.

**Caveat:** in the testbed the signal is planted *by regexes*, so a regex-compiling agent is favoured. Real notes are fuzzier.

## 5. Mayo rerun: strict cutoff, nested FIT/CAL/TEST, local readers only

Code: `research/mayo_strict_20261024/mayo_strict.py`. Aggregates are in `research/mayo_strict_20261024/out/`.

**Design:**
- **Strict note rule:** keep a bundle only if every note date is more than 2 calendar days before the scheduled day; otherwise blank the whole bundle.
  - 4,545 bundles were blanked.
  - 47.5% of cases keep notes.
- **Split:** TEST is the Metis forward-time hold-out (last 20%, patients disjoint). FIT and CAL are split by each patient's first case time, with no patient overlap.
- **Table model:** XGBoost trained on FIT. FIT residuals are 5-fold patient-grouped out-of-fold.
- **Certificates:** each patient is one unit, cap 120.

| Model (TEST, 2,599 cases) | Duration MAE (min) | Within-group R² | Certified (betting) | Risk AUC | Risk decile Spearman |
|---|---|---|---|---|---|
| Table only | 33.05 | 0 | — | 0.628 | 0.83 |
| + TF-IDF | 33.01 | 0.001 | 0 | 0.630 | 0.97 |
| + clinical concept rules | **32.96** | 0.003 | 0 | 0.629 | 0.92 |
| + rules, TF-IDF, meta, agent concepts (2 rounds) | 33.20 | −0.003 | 0 | 0.628 | 0.93 |
| 11 historical LLM features (lineage unverified) | 32.90 | 0.016 | 0.1 | 0.632 | 0.95 |

**Diagnostic: lenient cutoff, keeping notes ≤ 2 days before the case.** TF-IDF R² rises from 0.001 to 0.007, and the local stack from 0.002 to 0.011 (MAE 32.88).

**What Mayo says, plainly:**
1. **Under the strict rule, local note readers add about 0.1 min of MAE beyond the (already rich) table.** That is not certifiable at about 2,000 CAL patients.
2. **The agent's concepts passed FIT but failed on the forward-time TEST.** This is temporal shift plus small FIT gains, which is the protection working as intended.
3. **Most text signal sits in notes dated within 2 days of the case.** Whether those notes are admissible is a **data-policy decision**: it needs per-note timestamps. It is not a modelling decision.
4. **Risk forecast form matters more than the readers on Mayo.**
   - The empirical, family-specific residual forecast has Brier 0.226; the Gaussian form has 0.271.
   - The notes make the risk *ranking* more monotone (decile Spearman 0.83 → 0.93–0.97) at about the same AUC.

## 6. What changes, and next

- **Bound:** use betting (`tight_bounds.scalar_cert(method="betting")`) as the default scalar certificate. Declare the cap; tune λ and q on FIT.
- **Risk labeler:** use the residual forecast as the Metis default. Use the empirical family-specific error distribution on real data and the Gaussian on the testbed.
- **Information definition:** treat the table-conditioned symbolic sample space and the agent's concept refinements as the embedding-free representation. Certify each accepted refinement via the portfolio.
- **Mayo next steps:**
  - per-note timestamps, so bundles can be rebuilt from eligible notes only instead of blanking 42% of them;
  - local OpenMed entities on the strict notes;
  - agent acceptance on a *time-forward* FIT split, to stop temporally fragile concepts;
  - a larger CAL, or pooling years, for certificates.
