# Bound-guided risk-labeler development: LLM, OpenMed and Jev readers inside Metis (2026-10-20)

- **Code:**
  - `metis/reader_gain.py` (gate, refine, certify)
  - `research/instrumented_20261019/risk_labeler_gain.py` (MTSamples testbed)
  - `research/instrumented_20261019/llm_reader_mtsamples.py` (LiteLLM readers, run on the Mac)
- **Theory:** `docs/theory/INSTRUMENTED_INFORMATION_20261019.md`. This applies P1–P4 with the outcome replaced by the frozen model's **miss indicator**.

## The architecture

```text
note text ──► readers (Horos) ───────────────────────────────────────────────┐
              medspaCy rules (fast, free, local)                              │
              OpenMed NER: disease / pharma / anatomy / oncology / pathology  │  label-free features
              OpenMed PII (de-identification guard before any remote call)    │
              LLM via opmed LiteLLM (schema extraction; public or approved data)
              Jev decision model (pre-declared questions; OpenRouter, paused) │
                                                                              ▼
Pythia: GATE ── certified risk-information gain of each reader beyond the labeler's cells (portfolio, α/J)
          │
Metis:  REFINE ── accepted reader's bins become a new level of the risk hierarchy (outcome-blind)
          │
Pythia: CERTIFY ── local-risk v2 intervals + contrasts on the refined labeler (support partition, one α ledger)
          │
Pythia: GUARD ── reader/model upgrade regression test (P4); leakage audit (time-masked reader)
```

The bound is the **development objective**. A reader enters the risk labeler only if it carries certified information about the model's *errors* beyond the current risk cells. The labeler's sharpness (certified-higher / certified-lower share) is then certified, never just estimated.

## Testbed results

Setup: real MTSamples notes with a planted duration outcome, so the truth is known.

- **Deployed model:** table-only (specialty cell means), fitted on FIT notes.
- **Risk target:** W30 miss.
- **Rows:** 3,423 notes; FIT 1,711 and CAL 1,712, each patient in only one split.
- **True risk information** beyond the labeler cells: 0.0202.

### Step 1, GATE: certified risk-information gain, α/9 per reader

| Reader | Certified gain | Accepted |
|---|---|---|
| **Jev (5 declared surgical questions)** | **0.00028** | yes |
| TF-IDF + SVD | > 0, below 10⁻⁵ | yes (barely) |
| medspaCy rules; each OpenMed NER model; all OpenMed | 0 | no |

### Steps 2–3, REFINE and CERTIFY: share of CAL cases with a certified risk direction

Every certified direction was checked against the known true miss probability.

| Labeler | m = 100 | m = 300 | m = 500 |
|---|---|---|---|
| baseline (table cells only) | 0 | 0 | 0 |
| + Jev reader bins | 0 | 0 | **0.335** (574 high-risk cases, 0 wrong) |
| + TF-IDF reader bins | 0 | 0 | 0 |

### Reading the results

- **The gate works.** Only the reader that actually reads anesthesia and blood loss passes. The planted risk lives there: the true miss probability is 0.27 in the Jev reader's high-risk bin, against 0.15 in the other two bins.
- **The bound tells you how big a risk group must be.** With 1,712 calibration patients, a 7-point risk difference is only certifiable when regions hold about 500+ patients. At m = 100, many small regions each pay a large share of α.
- **The threshold must be chosen in advance.** The support threshold m must be declared before outcomes, from a width tolerance; picking the best m afterwards is not allowed. The grid above is reported as a sensitivity analysis.
- **Planning rule.** The half-width of a region's contrast is roughly √(2·ln(1/α_tail)·p(1−p)/m). For p ≈ 0.2, α_tail ≈ 10⁻³ and a target gap of 0.07, that needs m ≈ 550.
- **Implication for Mayo:** with about 5,000 calibration patients, a reader-refined labeler at m ≈ 500 can resolve risk splits of about 5–7 points.

## Running LLM readers through the opmed LiteLLM

The LiteLLM endpoint is internal (VPN), so it cannot be reached from Claude's sandboxes. Run it in your Mac's own Terminal, which is on the opmed network. The data is public MTSamples only.

```bash
cd ~/PycharmProjects/pythia-metis-horos/research/instrumented_20261019
export LITELLM_API_KEY=...            # or put it in ~/PycharmProjects/.litellm_key (never committed)
python3 llm_reader_mtsamples.py --list-models
python3 llm_reader_mtsamples.py --csv ~/PycharmProjects/_claude_transfer/llm_run/mtsamples.csv \
    --models <cheapest-model> --prompts extract_v2 --rows results/risk_rows.json --max-calls 3500 --workers 4
```

- This makes one call per note on the 3,423 testbed rows. Your LiteLLM budget is enforced server-side, and `--max-calls` is a second cap.
- The output, `readers.jsonl`, has no note text. Commit it, or tell Claude, and `risk_labeler_gain.py` adds the LLM reader to the gate and refinement automatically.

## Update 2026-10-23: LLM reader through the opmed LiteLLM (`gpt-6-luna`, prompt `extract_v2`)

Run from the Mac's Cowork sandbox over the internal LiteLLM, on public MTSamples notes only.

- **Coverage:** 3,416 of the 3,423 evaluation notes were read. 7 notes returned HTTP 400 and get an uninformative read.
- **Cost:** 3,457 calls, about USD 0.52 estimated from LiteLLM's listed prices. The exact figure is on the LiteLLM Usage page.
- **Fixes during the run:**
  - `max_tokens` was raised from 200 to 600, because hidden reasoning tokens truncated about 40% of answers at 200.
  - The runner now has a USD cap (`--max-usd`), flushes every line, and retries unparsed answers.

### GATE (this run's portfolio, α/3: TF-IDF, LLM, oracle reference)

| Reader | Certified risk-information gain |
|---|---|
| oracle concepts (reference ceiling) | 0.00298 |
| **LLM reader (gpt-6-luna)** | **0.00124** (42% of the ceiling) |
| TF-IDF + SVD | 0.00006 |

Earlier portfolio (α/9): Jev 0.00028; all OpenMed models and medspaCy 0. That portfolio used a different α split, so its numbers compare to these only roughly.

### REFINE + CERTIFY (share of CAL cases with a certified risk direction; checked against the known truth)

| Labeler | m = 100 | m = 300 | m = 500 |
|---|---|---|---|
| baseline (table cells only) | 0 | 0 | 0 |
| + LLM reader bins | 0 | 0 | **0.335** (574 cases certified higher-risk, 0 wrong) |
| + TF-IDF bins | 0 | 0 | 0 |

The oracle row is omitted: its predicted-probability bins tie, collapsing to 2 regions, which is a binning artefact.

### Reading

- **Gate:** the LLM reader recovers 42% of the certifiable risk information of the perfect-concept reference. That is about 20× bag-of-words.
- **Refinement:** at a support threshold that can resolve a 7-point risk gap, the LLM-refined labeler certifies a third of cases as higher risk, with no wrong directions. The baseline labeler certifies none.

**Data note:** the cloud workspace was reset on 2026-10-23. Its gitignored reader caches were lost: the Jev answers, the OpenMed entity caches and the medspaCy features. The LLM answers are safe in `readers.jsonl` on the Mac. From now on, reader caches are copied to the Mac after every run.
