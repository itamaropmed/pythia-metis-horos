# Open-source medical models in the Pythia / Metis / Horos pipeline (2026-10-19)

**Code:**
- `horos/represent/openmed_readers.py` (readers)
- `research/instrumented_20261019/run_openmed_readers.py` (batch runner, resumable)
- `tests/test_openmed_readers.py`

Everything runs **locally and free**: no note text leaves the machine. That makes these the readers to use on Mayo notes, where the remote LiteLLM is not approved.

## Where each piece plugs in

| Component | Model / tool | Role in the pipeline |
|---|---|---|
| **Fast reader** (Metis "cheap reader") | medspaCy: ConText negation, historical and hypothetical status; target rules | About 0.1 s per note on 2 CPUs. Gives concept × {asserted, negated, other} features. It is the always-on first read that Metis routes from. |
| **Entity reader** | OpenMed NER: DiseaseDetect, PharmaDetect, AnatomyDetect, OncologyDetect, PathologyDetect (TinyMed and SuperClinical sizes) | Entity bags become features, keeping only terms seen in ≥ 30–50 notes or patients. These are Pythia readers, so they enter the portfolio, the ledger and the regression test. |
| **Privacy guard** | OpenMed PII, e.g. `OpenMed-PII-SuperClinical-Small-44M-v1` | Three uses: (1) `--deidentify-with` in `llm_reader_mtsamples.py` de-identifies locally before any LiteLLM call; (2) `phi_vocabulary()` feeds `atlas.reportable_terms(exclude=…)`, so a name or MRN token is never reported, whatever its frequency; (3) `regex_phi_sweep()` is a zero-cost backstop. |
| **De-identified-text reader** | OpenMed PII → TF-IDF | Lets Pythia certify the **information cost of de-identification**: the same reader on original versus de-identified text, using the P4 regression test or a portfolio comparison. |

## Results (MTSamples, public notes, local CPU)

All five OpenMed TinyMed NER models (disease, pharma, anatomy, oncology, pathology; DistilBERT) run at about 0.2 s per note per model on 2 CPUs.

### A. Real target: the note's medical specialty (6 classes), beyond a weak table (note-length decile)

n = 1,475 notes; one α ledger over 8 readers. These are certified lower bounds on information, in Gini units.

| Reader | Features | Certified information |
|---|---|---|
| **all OpenMed NER models together** | 1,239 entity features | **0.455** |
| TF-IDF + SVD (bag of words) | 200 | 0.425 |
| OpenMed AnatomyDetect alone | 523 | 0.420 |
| OpenMed DiseaseDetect | 273 | 0.228 |
| OpenMed PathologyDetect | 219 | 0.181 |
| OpenMed OncologyDetect | 119 | 0.078 |
| OpenMed PharmaDetect | 105 | 0.071 |
| medspaCy generic surgical rules | 36 | 0.022 |

- The OpenMed entity readers together certify **more** information than bag-of-words.
- Every feature is a named clinical entity (interpretable, auditable), and they run free and local.
- Anatomy entities alone nearly match bag-of-words.

### B. Planted target (known truth 0.0179): anesthesia type, blood-loss number, complications

| Reader | Certified bound |
|---|---|
| oracle concepts (reference ceiling) | 0.0046 |
| TF-IDF + SVD | 0.0007 |
| medspaCy rules | 0.0000 |
| every OpenMed NER model, the combined reader, and their ensemble | 0.0000 |

The planted signal lives in anesthesia type and a blood-loss *number*. No disease, drug or anatomy model tags these, so the entity readers certify nothing. They correctly claim 0 rather than a false positive.

**Lesson (the alignment principle):** reader choice must follow what drives the outcome. For surgical duration, the relevant content is procedure, anesthesia and intra-operative events. That argues for a procedure/anesthesia extractor (rules, or an LLM reader with a schema) on top of OpenMed's disease, drug and anatomy readers. It also argues for running the portfolio rule (P2) across all of them, at α/J.

### C. De-identification cost (OpenMed PII → TF-IDF)

Running (resumable, slow on this 2-CPU workspace).

## Model weights

The five TinyMed NER models were downloaded on the Mac and copied from the local Hugging Face cache, because direct Hub downloads are blocked in the sandboxes. Earlier, only these OpenMed models were in the cache:

- `OpenMed-PII-SuperClinical-Small-44M-v1`, now in use;
- `privacy-filter-multilingual`.

To use the NER models, either:

1. allow `huggingface.co` (and `cdn-lfs.huggingface.co`, `*.xethub.hf.co`) in Claude's network settings; or
2. run once on your Mac:
   ```
   huggingface-cli download OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M
   huggingface-cli download OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-125M
   huggingface-cli download OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M
   ```

   Then `python3 run_openmed_readers.py ner --ner disease_tiny,pharma,anatomy` picks them up from the cache.

## Next

1. Run the OpenMed NER readers on MTSamples, then on Mayo locally (aggregates only), as Pythia readers.
2. Certify the de-identification cost on MTSamples, then on Mayo.
3. Metis routing: medspaCy (fast) → OpenMed NER (medium) → LLM (expensive), with the value-of-reading router from E9 and certified information per unit cost.

## Jev (TypeSafe, via OpenRouter): a fast, cheap probabilistic reader

`horos/represent/jev_reader.py` and `research/instrumented_20261019/run_jev_reader.py`.

**What Jev is:** a non-generative decision model. It answers only the questions we declare, over the options we declare. It returns a probability per option, with no free text, at about 0.3 s per call, priced per input token (fractions of a cent per note).

**Why it fits Pythia:**
- The question set is fixed before outcomes, so it is one pre-declared reader in the portfolio (P2).
- Its output is already a probability vector, so it is a natural input to the projection certificate.
- It cannot invent fields that were not asked for.

**Declared surgical questions:**
- anesthesia type (choice);
- blood-loss band (score);
- complication occurred (yes/no, with negated mentions counted as no);
- emergency (yes/no);
- procedure complexity (score).

These target exactly the content that the OpenMed disease, drug and anatomy readers missed on the planted surgical outcome.

**Privacy:** Jev is a remote API. Use it on public data, or after local OpenMed PII de-identification (`--deidentify-with`). Never use it on Mayo notes without approval.

**To run:**
1. Allow `openrouter.ai` in Claude's network settings.
2. Put an OpenRouter key in `~/PycharmProjects/.openrouter_key`. It is never committed.
3. Run `python3 run_jev_reader.py`, then `python3 mtsamples_certify.py`.

### Jev result (planted surgical target; run stopped at 3,423 of 4,999 notes at the owner's request, spend USD 0.20)

Same 3,423 notes for every reader. True information on this subset: 0.0202. Bounds are certified at α/5.

| Reader | Certified bound |
|---|---|
| oracle concepts (reference ceiling) | 0.0053 |
| **Jev, 5 declared surgical questions** | **0.0008** (4× bag-of-words) |
| TF-IDF + SVD | 0.0002 |
| all OpenMed NER models; Jev + OpenMed combined | 0.0000 |

- Jev's "general anesthesia" probability averages 0.995 when the planted concept is present and 0.12 when it is absent.
- The gap to the ceiling comes from definitional mismatch: the planted rule needs "general" inside an ANESTHESIA section and a blood-loss threshold of 100 ml.
- Combining Jev with 1,239 OpenMed features loses the signal: too many features for about 1,700 training rows per fold. A reader portfolio, which keeps the best reader, beats feature concatenation.
- **Budget rule from the owner:** no further OpenRouter spend. Future LLM readers go through the opmed LiteLLM budget only.
