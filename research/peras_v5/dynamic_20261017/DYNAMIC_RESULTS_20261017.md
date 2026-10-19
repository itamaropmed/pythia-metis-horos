# Dynamic geometric embeddings for AFI certificates (development, 2026-10-17)

- Theory: `docs/theory/AFI_TEXT_THEOREMS_20261017.md` (Theorems 1–5, Propositions 6–8).
- Code: `pythia/core/dynamic_geometry.py`.

## Geometries compared

The portfolio runs at α/4 on the generated data and α/5 on MTSamples, Bonferroni-charged.

| Geometry | How it is built | Outcome information used |
|---|---|---|
| static | TF-IDF (1–2-grams) → SVD(64) | none |
| diffusion | diffusion-map coordinates of the kNN graph at times t = 1, 4, 16, as a portfolio | none |
| posterior (sequential) | posterior probabilities from a classifier trained on *earlier batches*; batch fractions 20/40/40 | past batches only (Theorem 5) |
| NCA (sequential) | Neighbourhood Components Analysis projection learned on earlier batches | past batches only |

## 1. Ten generated datasets (n = 8000, 5 runs each, exact truth)

Each cell shows the mean certified ΔI(T|X), then "(runs > 0 / violations)".

| Dataset | True ΔI | static | diffusion | posterior | NCA | **portfolio** |
|---|---|---|---|---|---|---|
| D01 echo (truth 0) | 0 | 0 (0/0) | 0 (0/0) | 0 (0/0) | 0 (0/0) | **0 (0/0)** |
| D02 novel findings | 0.078 | 0.025 (5/0) | 0.025 (5/0) | 0.007 (5/0) | 0.018 (5/0) | **0.028 (5/0)** |
| D03 negation | 0.084 | 0.017 (5/0) | **0.027 (5/0)** | 0.012 (5/0) | 0.019 (5/0) | **0.027 (5/0)** |
| D04 synergy | 0.093 | 0.032 (5/0) | **0.044 (5/0)** | 0 (0/0) | 0.006 (3/0) | **0.044 (5/0)** |
| D05 boilerplate | 0.079 | 0 (0/0) | 0.002 (3/0) | 0.001 (1/0) | 0.003 (2/0) | **0.005 (4/0)** |
| D06 partial mentions | 0.036 | 0 | 0 | 0 | 0 | 0 (below the limit) |
| D07 number in text | 0.051 | 0 | 0 | 0 | 0 | 0 (below the limit) |
| D08 outcome leak | 0.126 | 0 (0/0) | 0 (0/0) | **0.033 (5/0)** | 0.016 (5/0) | **0.033 (5/0)** |
| D09 4-class | 0.074 | 0.003 (3/0) | **0.021 (5/0)** | 0.003 (3/0) | 0.011 (5/0) | **0.021 (5/0)** |
| D10 style confound | 0.019 | 0 (0/0) | 0 (0/0) | 0 (0/0) | 0 (0/0) | 0 (0/0) |

**Reading**
- **Validity.** 0 violations in 250 certificates.
- **No false information.** None on the echo or style-confound datasets.
- **The two dynamic geometries do different jobs.**
  - Diffusion time is the best geometry for structured signal: negation, synergy and multi-class.
  - The sequentially learned posterior geometry is the only one that finds rare decisive tokens, such as the outcome-leak word.
  - The portfolio certifies 6 of the 9 datasets that carry text-beyond-table signal in at least 4 of 5 runs (D02, D03, D04, D05, D08, D09). The static multilevel AFI from the previous run reached that rate on only 3 of them (D02, D03, D04). The three still missed (D06, D07, D10) sit below the resolution limit at this n.

## 2. Real: MTSamples, extraction-completeness ladder with a neural embedding

The e5-small-v2 weights were taken from your local Hugging Face cache and run in the sandbox. The notes are public MTSamples, so no private data was used. Values are the mean of 3 seeds.

| Outcome | Table | TF-IDF static | e5 static | e5 diffusion | posterior (seq.) | NCA (seq.) | **Portfolio** |
|---|---|---|---|---|---|---|---|
| Specialty (K = 11) | note type | **0.259** | 0.219 | 0.195 | 0.198 | 0.152 | **0.259** |
| | + age, sex | **0.130** | 0.109 | 0.102 | 0.082 | 0.068 | **0.130** |
| | + length | **0.089** | 0.074 | 0.060 | 0.050 | 0.038 | **0.089** |
| Surgery vs not | note type | **0.120** | 0.105 | 0.099 | 0.076 | 0.096 | **0.120** |
| | + age, sex | **0.061** | 0.053 | 0.057 | 0.046 | 0.049 | **0.061** |
| | + length | **0.045** | 0.043 | 0.045 | 0.033 | 0.036 | **0.045** |

Text-only information lower bound, with no table: specialty TF-IDF 0.449 vs e5 0.388; surgery 0.341 vs 0.326.

**Reading**
- **The neural embedding is valid but less informative than TF-IDF here.** Specialty is carried by domain vocabulary, which sparse n-grams resolve better than a general-purpose 384-dimensional sentence embedding.
- **The learned dynamic geometry does resolve more signal.** Its point contrasts in later batches are up to about 2× the static ones (for example 0.23 vs 0.10 at the finest table). But at n = 2,357 the certificate loses that gain to batch splitting: table cells fragment inside batches, and each batch pays its own deviation term.
  - A variant with a 30% warm-up batch and the other 70% certified was worse (0.04–0.13).
  - Dynamic geometry pays off at larger n, as in the generated data at n = 8000.
- **Practical rule.** Use a portfolio. The certificate automatically keeps the best geometry, and the Bonferroni cost is small.

## 3. MIMIC

MIMIC-IV / MIMIC-IV-Note is not reachable from this sandbox: PhysioNet is blocked and access is credentialed. It is also not present in the connected folders on your machine.

`scripts/run_vtc_mimic.py` is ready to run **locally** by a credentialed user, and writes aggregates only:
- 30-day readmission from discharge summaries;
- in-hospital mortality from radiology reports charted in the first 24 hours (discharge summaries leak death).

The table is age band × sex × admission type × insurance, with one admission per patient. The script ran end to end on a mock MIMIC-shaped dataset.
