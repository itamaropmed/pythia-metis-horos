# Agreement-Field Information: ten generated datasets and one real dataset (development, 2026-10-17)

Theory: `docs/theory/AGREEMENT_FIELD_INFORMATION_20261017.md`.

**Setup**
- All statements are simultaneous at α = 0.05 (Bonferroni over 6).
- Text geometry: unigram+bigram TF-IDF → SVD(64), unsupervised.
- Generated data: n = 8000 per run, 15 runs per dataset.

## 1. Ten generated table+text datasets with exact laws

**How the data are built**
- Every dataset has its own pseudo-vocabulary and writing style: length, boilerplate, phrase shuffling, abbreviation rate. The ten are therefore out of distribution with respect to one another.
- Table: age band × sex × lab band (24 cells).
- Text: latent findings rendered as positive mentions or negations, plus noise. The posterior p(Y | table, note) is exact.

**Reading the table**
- "Cov" counts runs where the two-sided interval covered the truth.
- "Pos" counts runs with a certified ΔI > 0.
- "Viol" counts runs where a bound crossed the truth.
- "AV" is the model-based added value, the Brier score of logistic regression on the table minus logistic regression on table + text. It is an estimate, not a bound.

| Dataset | Mechanism | True I(X) | I(X) cov | True ΔI(T\|X) | Certified ΔI (pos / viol) | H(Y\|X) cov | H(Y\|X,T) upper viol | p < 0.05 | Design co-info | Model AV (runs < 0) |
|---|---|---|---|---|---|---|---|---|---|---|
| D01 | Text echoes the table; Y from the table only | 0.079 | 15/15 | 0 | 0.000 (0/0) | 15/15 | 0 | 0 | **+0.062** (echo) | −0.018 (15) |
| D02 | Novel findings in text | 0.019 | 15/15 | 0.079 | 0.014 (14/0) | 15/15 | 0 | 15 | +0.013 | +0.049 (0) |
| D03 | Negation ("denies X" vs "reports X") | 0.018 | 15/15 | 0.084 | 0.008 (12/0) | 15/15 | 0 | 15 | +0.016 | +0.060 (0) |
| D04 | Synergy (sex × finding, age × finding) | 0.035 | 15/15 | 0.092 | **0.023 (15/0)** | 15/15 | 0 | 15 | **−0.069** (synergy) | **−0.022 (15)** |
| D05 | Heavy boilerplate (~220 noise tokens) | 0.018 | 15/15 | 0.079 | 0.000 (0/0) | 15/15 | 0 | 0 | +0.014 | +0.030 (0) |
| D06 | Findings mentioned only half the time | 0.018 | 15/15 | 0.036 | 0.000 (0/0) | 15/15 | 0 | 3 | +0.007 | +0.009 (0) |
| D07 | Lab value written as a number in text | 0.065 | 15/15 | 0.051 | 0.000 (0/0) | 15/15 | 0 | 0 | +0.017 | +0.020 (0) |
| D08 | Outcome leak (text reveals Y 30% of the time) | 0.080 | 15/15 | 0.126 | 0.000 (0/0) | 15/15 | 0 | 1 | +0.029 | +0.075 (0) |
| D09 | 4-class outcome | 0.163 | 15/15 | 0.074 | 0.002 (5/0) | 15/15 | 0 | 15 | 0.000 | +0.051 (0) |
| D10 | Writer style tied to table cell, plus a weak finding | 0.071 | 15/15 | 0.019 | 0.000 (0/0) | 15/15 | 0 | 0 | **+0.071** (echo) | +0.004 (1) |

**Reading**
- **Validity.** Across 150 runs:
  - the two-sided table statements covered the truth every time;
  - the certified text-beyond-table bound never exceeded the truth;
  - the joint-floor upper bound was never violated.
- **No false information.**
  - The echo dataset and the style-confound dataset never produce a false certificate. The text predicts the table there, but within-cell anchoring removes that.
  - The design co-information flags both as echo (+0.06, +0.07).
- **Synergy is detected, and the model misses it.** In D04 the certificate is positive in 15/15 runs and the co-information is negative, while logistic regression reports *negative* added value in 15/15 runs.
- **Power is the limit, and the geometry decides it.**
  - With unsupervised SVD geometry, signal carried by a *single rare token* is not resolved: boilerplate (D05), a number (D07), a leak word (D08), partial mentions (D06). The nearest notes are near only in their noise.
  - The minimum certifiable ΔI at n = 8000 is about 2·√(log(6·levels/α)/n) ≈ 0.05, plus the drift the geometry fails to remove.
  - A cross-fitted learned geometry (next section) is the fix under test.

## 2. Learned geometry (cross-fitted; still outcome-blind for the paired rows)

**Learned metric.** A logistic model is trained on the *other* fold. It selects the 300 most informative n-grams and weights them, and those weighted coordinates become the pairing geometry. The geometry portfolio is SVD at α/2 and the learned metric at α/2, with folds at α/4. Run counts are 4–10 per dataset (fewer for D04, D07 and D08 because of runtime).

| Dataset | True ΔI | SVD only (pos) | Learned only (pos) | **Portfolio** (pos / viol) |
|---|---|---|---|---|
| D01 echo (truth 0) | 0 | 0 (0/5) | 0 (0/5) | 0 (0/5, 0 viol) |
| D08 outcome leak | 0.126 | 0 (0/5) | **0.040 (5/5)** | **0.040 (5/5, 0)** |
| D05 boilerplate | 0.079 | 0 (0/5) | 0.003 (4/5) | 0.003 (4/5, 0) |
| D02 novel findings | 0.078 | 0.014 (5/5) | 0.000 (2/5) | 0.014 (5/5, 0) |
| D04 synergy | 0.093 | 0.024 (4/4) | 0 (0/4) | 0.024 (4/4, 0) |
| D06 partial mentions | 0.036 | 0 | 0 | 0 (below the sample-size limit) |
| D07 number in text | 0.051 | 0 | 0 | 0 (below the limit once folds halve n) |

**Reading.** The two geometries complement each other.
- The learned metric finds rare decisive tokens: the leak word, or the signal phrase inside boilerplate.
- The unsupervised geometry keeps the broad and interaction signal that a linear selector misses.
- The portfolio never certified falsely.
- Signals near 2·√(log(6·levels/α)/n) are not certifiable at this n. That is the sample-complexity price of a guarantee.

## 3. Real data: MTSamples, extraction-completeness certification

The "table" is a structured extraction from each note: note type from section headers, age band, sex and length band. This is what a clinical NLP pipeline would populate. The outcome is the real specialty. Results are averaged over 5 design seeds.

| Outcome | Table (extraction ladder) | Cells | I(X) table info [L, U] | H(Y\|X) table floor [L, U] | **Certified text info beyond table** | H(Y\|X,T) upper | I(T) text info ≥ | p (text adds nothing) | Model AV |
|---|---|---|---|---|---|---|---|---|---|
| Specialty (K = 11) | none | 1 | — | [0.73, 0.82] | **≥ 0.453** | 0.272 | 0.448 | < 0.05 (5/5) | 0.381 |
| | note type | 5 | [0.08, 0.27] | [0.55, 0.65] | **≥ 0.254** | 0.296 | | < 0.05 | 0.210 |
| | + age, sex | 55 | [0.11, 0.59] | [0.22, 0.61] | **≥ 0.118** | 0.362 | | < 0.05 | 0.194 |
| | + length band | 113 | [0.11, 0.61] | [0.20, 0.59] | **≥ 0.075** | 0.395 | | < 0.05 | 0.199 |
| Surgery vs not (K = 2) | none | 1 | — | [0.44, 0.53] | **≥ 0.347** | 0.090 | 0.342 | < 0.05 | 0.384 |
| | note type | 5 | [0.14, 0.32] | [0.21, 0.30] | **≥ 0.115** | 0.091 | | < 0.05 | 0.153 |
| | + age, sex | 55 | [0.15, 0.50] | [0.04, 0.30] | **≥ 0.057** | 0.126 | | < 0.05 | 0.155 |
| | + length band | 113 | [0.14, 0.50] | [0.04, 0.31] | **≥ 0.040** | 0.154 | | < 0.05 | 0.168 |

**Reading**
- **Extraction completeness, certified.** Even with note type, age, sex and length extracted, the free text still carries **at least 0.075** Brier information about specialty, and at least 0.040 about surgery, beyond the structured fields. The p-value is < 0.05 in every seed.
- **Each extracted field removes certified residual signal.** The residual falls along the ladder: 0.45 → 0.25 → 0.12 → 0.08.
- **Echo grows as fields are extracted from the text itself.** Co-information rises from 0.20 to 0.35. This is expected, because the table is a function of the note.
- **Weakness.** Finer tables have small cells (as few as 4 rows), so the two-sided table intervals widen through the O(1/m) term. A cell-size-weighted design would tighten them.
