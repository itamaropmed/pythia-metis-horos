# Improving the bound and the risk predictor; information-theoretic reading; the agentic reader loop (2026-10-23)

**Scope**
- Public MTSamples notes and known-truth simulations only. No Mayo data was used.
- Layers stay separate:
  - **Pythia** certifies; it trains nothing.
  - **Horos** learns scores and runs readers.
  - **Metis** forecasts per-case risk and runs the agent.

**Status:** development experiments with known truth. Nothing here is preregistered.

**Code**

| Component | File |
|---|---|
| Horos score learning: variational scores, CV ridge, scalar scores, Rayleigh stacking | `horos/represent/score_learning.py` |
| Torch-free OpenMed inference | `horos/represent/numpy_ner.py` |
| Pythia: scalar projection bound (Chernoff + empirical Bernstein), score taming | `pythia/core/instrumented.py` |
| Metis agent: diagnose, propose, probe, score, budget ledger | `metis/agent/reader_search.py` |
| Testbeds and experiments | `research/instrumented_20261019/`: `testbed_v2.py`, `reader_features.py`, `run_bound_improvements.py`, `run_bound_gbm.py`, `metis_risk_v2.py`, `continuous_risk_gate.py`, `agent_*.py`, `run_numpy_ner.py`, `run_ner_vm.py` |
| Tests | `tests/test_score_learning_and_agent.py` (+ existing instrumented and reader-gain tests): 7 passed |

## 1. Information-theoretic view of what a note says (not IE/RE)

The quantity is the **conditional information the note carries about the outcome, or about the model's error, beyond the table**:

  V = E | P(Y | X, T) − P(Y | X) |²

This is the Brier/χ² form of I(Y; T | X). Its variational form is:

  V = sup over h with E[h | X] = 0 of  (E⟨e_Y − p̄, h⟩)² / E|h|²

The optimal h is the *information direction* h\* = P(Y | X, T) − P(Y | X). So "extracting information" means estimating that direction from any reader's output. Pythia then certifies how much of V the estimated direction captures. The certified bound equals alignment² × V minus a deviation term. Every reader (LLM, OpenMed, Jev, bag-of-words) is judged by one number: how well it aligns with h\*.

## 2. Improving the bound

### 2a. Score learning and stacking

Setting: known-truth simulation, 30 replicates, α = 0.05. Each value is the certified share of the true information.

| Scenario | n | Classifier probabilities, best reader (α/J; previous default) | Variational ridge, best reader | Variational on concatenated features | **Variational Rayleigh stack** |
|---|---|---|---|---|---|
| 3 complementary readers + junk | 2,000 | 0.128 | 0.118 | 0.181 | **0.204** |
| | 8,000 | 0.239 | 0.219 | 0.341 | **0.352** |
| 5 noisy reads of the same note | 2,000 | 0.088 | 0.081 | 0.181 | **0.224** |
| | 8,000 | 0.186 | 0.171 | 0.394 | **0.423** |

The bound never exceeded the truth in any run.

**Stacking fits only J weights:** w ∝ G⁻¹c, the closed-form maximiser of (w′c)²/(w′Gw), computed on FIT out-of-fold scores. It **roughly doubles the certificate relative to the "best reader" portfolio**, because it combines complementary readers without over-fitting the way concatenation can.

### 2b. Nonlinear (gradient-boosted) scores

| n | Ridge stack | Gradient-boosted stack | Ridge, concatenated | Gradient-boosted, concatenated |
|---|---|---|---|---|
| 2,000 | 0.208 | 0.188 | 0.176 | 0.060 |
| 8,000 | 0.344 | 0.340 | 0.335 | 0.256 |

Nonlinear scores did **not** help at these sizes. Keep ridge stacking as the default.

### 2c. Continuous risk target

The tested target was a capped absolute error with a scalar projection bound, combining Chernoff with an empirical-Bernstein inequality and winsorised score taming. On testbed v2 it certified 0, even for the oracle reader. The binary miss target certified 25% of the truth for the oracle.

The cause is the range-based deviation terms: they are dominated by extreme score values relative to the small conditional variance of the error. A variance-adaptive inequality with heterogeneous ranges remains open. The code is kept, but no claim is made for it.

## 3. Improving the risk predictor (testbed v2, known truth)

**Testbed v2**
- Real MTSamples notes with a planted surgical duration.
- The duration is driven by **seven kinds of note content**: anesthesia type and blood loss (LLM/Jev territory), diabetes, hypertension, obesity, malignancy (OpenMed disease) and anticoagulants (OpenMed pharma).
- The deployed model is table-only.
- The risk target is W30 miss, with exact P(miss | case) known.
- Fixed outcome-free split of 3,423 rows: FIT 1,331, CAL 1,023, TEST 1,069.

### Per-case forecast accuracy on TEST

| Forecast | MAE vs true risk | Brier |
|---|---|---|
| Metis bins (table cell rate) | 0.0849 | 0.1901 |
| Gradient-boosted classifier, table only | 0.0857 | 0.1891 |
| Gradient-boosted classifier, table + any text reader | 0.10–0.17 (**worse**: over-fits 1,331 rows) | 0.19–0.23 |
| **Residual ridge, + LLM reader** | **0.0814** | **0.1877** |
| Residual ridge, + OpenMed disease / pharma / anatomy | 0.0845–0.0856 | ≈ 0.190 |
| Residual ridge, + stack of all real readers | 0.0840 | 0.1890 |
| Residual ridge, + oracle concepts (reference) | 0.0736 | 0.1850 |

Residual ridge forecasts the cell rate plus a × the stacked variational score, with a fitted on FIT out-of-fold scores.

**Gate (Pythia, CAL, α/7):** certified risk information beyond the table cells, against a truth of 0.0351.

| Reader | Certified | Share of truth |
|---|---|---|
| LLM reader | 0.00111 | 3% |
| OpenMed anatomy | 0.00014 | — |
| OpenMed disease, OpenMed pharma, TF-IDF | 0 | 0 |
| Stack of all real readers | 0.00092 | — |
| Oracle (reference) | 0.0097 | 28% |

**Certified layer (forecast bins, m = 250 declared in advance, audited on TEST truth):**
- Only the oracle-driven labeler certified directions: 33% of CAL cases, 0 wrong.
- No real-reader labeler certified anything at this calibration size.

### Reading

1. **The flexible risk model loses.** At this sample size, adding text to a flexible classifier makes the per-case risk forecast worse. The low-variance "table rate + certified-direction" forecaster is the one that improves on Metis bins, by 4% with the LLM reader.
2. **OpenMed entity bags carry almost no usable risk direction here, despite the planted comorbidity and anticoagulant effects.** Hundreds of sparse entity indicators, 1,331 training rows and two-sided miss noise mean the direction cannot be estimated. The bottleneck is sample size and the dimension of the reader, not the reader's recall.
3. **The gap to the oracle (28% vs 3%) is the room for better readers.** That gap is what the agent loop is for.

## 4. The agentic reader loop (Metis)

**Loop:** DIAGNOSE → PROPOSE → PROBE → SCORE → ACCEPT/REJECT.
1. **Diagnose:** FIT out-of-fold residuals of the current risk model, then the note terms most associated with them.
2. **Propose:** an LLM (opmed LiteLLM) turns those terms into at most 5 typed questions in Jev format, schema-validated. It sees no outcomes. Rejected question sets are fed back so they are not repeated.
3. **Probe:** a reader answers the questions on FIT notes only.
4. **Score:** the cross-fitted gain in plug-in signal, with versus without the new reader, on FIT only.
5. **Accept or reject:** only accepted sets are read on CAL/TEST. Selection never touches CAL, and the portfolio rule would keep certificates valid even if it did.

### Round results

| Round | Proposal (gpt-5.4-mini) | Agreement with planted concepts (diagnostic only) | FIT score | Decision |
|---|---|---|---|---|
| q1 | adhesions / scar tissue; displaced anatomy | ≤ 0.08 | negative on 600 probe notes | rejected (correctly irrelevant) |
| q2 | obesity severity; anticoagulation / bleeding risk; prior surgery; intra-operative injury; urgency | anticoagulants **0.56**, obesity 0.21 | +3×10⁻⁵ on all 1,331 FIT notes, 2 of 5 seeds positive | rejected (gain not distinguishable from noise) |

The agent found the right content: q2 recovers the planted anticoagulant signal. But a FIT-only score with 1,331 binary outcomes is too noisy to confirm a gain of this size. That points to the next fixes:
- a larger FIT set;
- a lower-variance score that uses the new reader's agreement with the *existing* residual direction;
- or accepting on a pre-declared prior combined with Pythia's portfolio rule on CAL.

### Jev

- **Question set q0 (the declared surgical questions):** OpenRouter returned *402 Payment Required* after 216 notes. The credit is exhausted; this session spent USD 0.011.
- **The agent's PROBE step therefore used the LiteLLM model as a Jev-style typed answerer:** probabilities per option, the same feature format.
- **To use Jev itself,** top up the OpenRouter credit. `agent_jev.py` runs any question set under the global USD cap.

## 5. Spend

| Service | This session | Earlier |
|---|---|---|
| LiteLLM | about USD 0.53, estimated from list prices (gpt-6-luna answers + 2 proposals) | about USD 0.52 (`extract_v2` run) |
| OpenRouter / Jev | USD 0.011 (402 after 216 calls) | USD 0.196 (answers lost in the cloud reset) |

**Cache safety:** all reader caches are now mirrored to the Mac (`results/reader_cache/`):
- NER for disease, pharma and anatomy;
- `llm_qa.jsonl`;
- `jev_qsets.json`;
- `readers.jsonl`.

## 6. What this changes, and what is next

- **Bound:** use variational Rayleigh stacking of readers as the default; it is valid and about 2× tighter than taking the best reader. Next: a variance-adaptive inequality with heterogeneous ranges, which unlocks continuous risk targets.
- **Risk predictor:** use residual-ridge on certified directions, not flexible models on raw text features, at clinical-cohort sizes.
- **Readers:** compress OpenMed entity bags into low-dimensional directions before they enter (supervised on FIT, or ontology roll-up). Prefer agent-proposed typed questions, which are low-dimensional and aligned by design.
- **Agent:** fix the scoring power, as above. Run Jev once credit is available, under the same ledger.
- **Mayo:** only after the provenance fixes in the red-team report: the conservative note cutoff, enforced lineage, and nested FIT/CAL/TEST. Mayo can use only local readers (OpenMed, rules) unless de-identified LiteLLM use is approved.
