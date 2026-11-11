# Instrumented information: simulations, weak spots and the research gap (2026-10-19)

- **Theory:** `docs/theory/INSTRUMENTED_INFORMATION_20261019.md`.
- **Code:** `pythia/core/instrumented.py`, `research/instrumented_20261019/` (`sim_lib.py`, `run_experiments.py`, `mtsamples_certify.py`, `llm_reader_mtsamples.py`).
- **Status:** a development suite with a known law. Nothing here is preregistered.
- **Sampling assumption:** these original runs generated independent rows. They do not validate repeated-patient dependence; see the later cluster-aware projection implementation and cluster-null regression test.

## Simulator

The simulated note says six binary "concepts"; four of them move the outcome, and two interact.

- The table has 20 cells, and concepts correlate with the cell.
- The truth p_i = P(Y | cell, concepts) is exact, so every certified quantity has a known true value.

A simulated **LLM reader** is defined by:
- the concepts its prompt asks about;
- its per-concept error rate;
- the number of hallucinated junk features it adds;
- optionally, leakage of a post-outcome note.

Readers' probabilities are cross-fitted logistic models. Level α = 0.05 throughout. Each figure is the mean over 15–100 replicates.

## Results

### E1. Validity and tightness

Tightness = certified lower bound ÷ true information beyond the table.

| Reader | n = 1,000 | 4,000 | 16,000 | Bound above truth |
|---|---|---|---|---|
| strong (all concepts, 5% error) | 0.19 | 0.44 | 0.60 | 0 / 1,350 |
| cheap (3 concepts, 15% error) | 0.06 | 0.18 | 0.28 | 0 |
| weak paraphrase (1 useful concept, 25% error) | 0 | 0 | 0 | 0 |
| hallucinating | 0 | 0 | 0 | 0 |

- **Under the simulated independent-row law:** a bad reader gives 0 in these runs. This is not a guarantee without the declared independent sampling-unit assumption.
- **Weak spot W1 — looseness.** At realistic n (4k), the bound certifies only 18–44% of the true information. It converges slowly: 60% at 16k.

### E2. Reader hacking on a null text

The text carries no information; K random prompt/model configurations are tried (20 replicates).

| Configurations K | Naive "any config certifies", each at α | Portfolio rule, α / K |
|---|---|---|
| 1 | 0% | 0% |
| 10 | 10% | 5% |
| 50 | 0% | 0% |
| 200 | **45%** | 0% |

What the portfolio costs when the text *is* informative (certified ÷ truth):

| K | 1 | 10 | 100 | 1,000 | 10,000 |
|---|---|---|---|---|---|
| certified ÷ truth | 0.46 | 0.42 | 0.34 | 0.29 | 0.27 |

Searching 10,000 configurations keeps 58% of the single-reader certificate. Reader choice becomes a cheap, safe search instead of a source of false findings.

### E3. Certified additive ledger

Cheap-reader bins first, then the strong reader for what those bins miss:

| Part | Bound | Truth |
|---|---|---|
| captured by the cheap reader's categories, B(a) | 0.0100 | 0.0193 |
| beyond those categories, V(X, a) | 0.0090 | 0.0258 |
| **ledger total** | **0.0190** | 0.0451 |
| strong reader alone | 0.0225 | 0.0451 |

- Both parts are valid: 0 misses in 100 replicates.
- The ledger is slightly weaker in total than the best single reader, because of the α split. What it adds is **attribution**: how much of the information the cheap reader holds, and how much it leaves on the table.

### E4. Reader regression test (an upgrade that drops a concept)

| n | Detection, lost concept | False alarm, superset upgrade |
|---|---|---|
| 2,000 | 100% | 0% |
| 8,000 | 100% | 0% |
| 32,000 | 100% | 0% |

This is the strongest result: certified "this model upgrade lost information", with no false alarms.

### E5 and E9. Selective reading

A cheap reader runs on every case; the expensive reader runs on a share q.

| Share read | Entropy rule (E5) | Learned value-of-reading rule (E9) |
|---|---|---|
| 0 | 0.17 | — |
| 10% | 0.20 | 0.24 |
| 25% | 0.28 | 0.31 |
| 50% | 0.39 | 0.42 |
| 100% | 0.54 | 0.54 |

Values are certified information as a fraction of the truth. The learned router reads half the notes and keeps 77% of the full-reading certificate. **Weak spot W2:** routing gains are modest, because the router only sees cheap features. This is the Metis opportunity.

### E6. Leakage from a post-outcome note

The estimand is the information in pre-outcome text.

| Leak rate | Bound above pre-outcome truth | Unmasked − time-masked bound |
|---|---|---|
| 0 | 0% | 0.0001 |
| 5% | 0% | 0.0052 |
| 20% | **39%** | 0.0253 |

- **Weak spot W3:** the certificate is only as good as the "text is pre-outcome" attestation.
- The time-masked comparison exposes leakage. The P4 regression test turns that comparison into a certified leakage audit.

### E7. Double dipping on a null text

With 40 hallucinated features and an over-fitted model:

| Training | False certificates |
|---|---|
| cross-fitted | 0 / 100 |
| fitted on the evaluation fold | **100 / 100** |

The held-out attestation is non-negotiable. A reader fine-tuned on the evaluation labels breaks everything. This matters for LLMs that may have seen public benchmark data.

### E8. Tightening levers (n = 4,000)

| Lever | Tightness |
|---|---|
| single noisy reader (20% error) | 0.11 |
| ensemble of 3 independent reads | 0.27 |
| ensemble of 9 independent reads | **0.43** |
| 2 folds (default) | 0.45 |
| 5 folds (α split, smaller evaluation folds) | 0.26 |
| 10 folds | 0.12 |

- **Prompt/model ensembles roughly quadruple the certificate** for noisy readers. Averaging independent reads removes reader noise from the score, and that noise only inflates the denominator.
- More folds hurt.

### MTSamples (real notes, planted binary outcome, 4,999 notes)

| Reader | Certified bound (α/J) | True information |
|---|---|---|
| TF-IDF + SVD | 0.0012 | 0.0179 |
| oracle concepts (reference ceiling) | 0.0058 | 0.0179 |
| LLM readers | **pending: the LiteLLM run on the Mac** | 0.0179 |

The question for the LLM run: does an LLM reader close the gap between bag-of-words (7% of the truth) and the oracle (32%)?

## Weak spots, ranked

1. **Looseness (W1).** The bound equals alignment² × V minus the deviation. Fixes:
   - reader ensembles (shown);
   - better nuisance models (gradient boosting on reader outputs);
   - a variance-adaptive (empirical-Bernstein-type) heterogeneous Chernoff bound;
   - pooled cross-fitting (open).
2. **Attestation risk (W3, E7).** Leakage and double dipping produce confident wrong certificates. The masking audit and the fine-tuning provenance check must be part of every run, not optional.
3. **Routing power (W2).** A router that sees only cheap features captures little extra.
4. **Lower-only.** Readers never give an upper bound. Pair the reader bound with the table-floor box (`pythia-cert nli`) for a two-sided statement.

## Where the world is, and the gap

- **Surrogate-label inference** (PPI, DSL, CDI; Angelopoulos et al. 2023, Egami et al. 2023):
  - needs gold labels and MCAR sampling;
  - makes prompt and model choice a source of error. "LLM hacking" (2025) reports wrong conclusions in about 31% of hypotheses.
  - The TEE framework (2026) shows confidence intervals that under-cover as n grows.
- **Variable-importance inference** (Floodgate; VIM/LOCO; GCM):
  - Floodgate's original model-X route assumes known P(T | X); the paper also gives a doubly-robust extension. Its guarantees are asymptotic, and it targets the same squared-loss mMSE gap. The extension must be included in matched comparisons before making a gap claim.
  - VIM/LOCO are asymptotic.
  - GCM is a test without an effect size.
  - None handles a *portfolio of readers*, an additive ledger, or upgrade regression.
- **Clinical LLM feature papers** report AUC gains from LLM-extracted note features, with no beyond-table certificate, leakage audit or reader-multiplicity control.
- **Budget-aware routing for clinical text** (2026) is empirical, with no guarantee on the information retained.

**Proposed framework: instrumented information accounting.** The projection inequality is finite-sample under its boundedness, independent-cluster, centering, and label-free-score assumptions. The present API does not authenticate those assumptions or precommitment, so a returned number is conditional until a trusted evidence boundary exists. Portfolio selection control applies only to a fixed finite family. The additive ledger (P3) and routing claims require separate theorem and implementation audits. Do not describe the current implementation as hacking-proof or as a new field before matched comparisons and independent review.
