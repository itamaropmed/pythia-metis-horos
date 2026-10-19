# Replicate certificates: known-signal tests on real clinical text and tabular data (2026-10-15)

**Status: development.** Nothing here is pre-registered. The next step is to freeze the code and registration, then confirm on untouched seeds.

**Theory:** `docs/theory/REPLICATE_PAIR_CERTIFICATES_20261015.md`. In the tables, values are the bound divided by the known floor (mean over runs). "Violations" counts runs where a bound exceeded the floor.

**Methods compared**
- **Accepted route:** min-count / hard-label averaging. Its concentration is the best of:
  - instance Hoeffding (Ishida et al.);
  - cell-level Bennett;
  - row-level domination via Chernoff–KL.

  It is reported as a single strongest baseline.
- **New routes:**
  - pair certificate (Theorem A);
  - linear unbiased minorant (Theorem C);
  - second-order slack (Theorem D);
  - invariance audit and within-cell independence audit.

## 1. Clinical text plasmode: MTSamples notes, nuisance copies

**Setup**
- 2,357 unique real notes serve as bases. Each has a planted label law (binary, or K = 4) driven by:
  - concepts;
  - an out-of-dictionary 'smok' channel;
  - length;
  - dose counts;
  - base-level noise.
- Copies (1–8 per base) are re-issued with nuisance edits:
  - name/MRN/date header;
  - re-dated text;
  - abbreviations swapped;
  - sections reordered;
  - case and whitespace changes.
- 40/60 development/certification split by base; about 3,500 certification rows; 12 runs per cell of the table.

| Scenario | K | Raw exact duplicates | Accepted (canonical cells) | Pair | Linear minorant | **Pythia** | 1-NN TF-IDF estimate (not a bound) |
|---|---|---|---|---|---|---|---|
| T1 nuisance-only | 2 | 0.04 | 0.58 | 0.52 | 0.61 | **0.61** | 0.84 |
| T1 | 4 | 0.04 | 0.59 | 0.54 | 0.62 | **0.62** | 0.81 |
| T2 copy-forward addenda, pairwise TV ≤ 0.06 declared | 2 | 0.01 | 0.04 | 0.51 | 0.56 | **0.57** | 0.88 |
| T2 | 4 | 0.01 | 0.26 | 0.53 | 0.55 | **0.55** | 0.85 |
| T3 canonicalizer wrongly drops dose numbers | 2 | 0.03 | 0.60 | 0.54 | 0.64 | **0.64** | 0.88 |
| T3 | 4 | 0.03 | 0.63 | 0.57 | 0.66 | **0.66** | 0.85 |

**Validity**
- Violations: 0/72 for every certificate.
- Pythia is strictly above the accepted route in 71/72 runs.
- The within-cell independence audit never rejected.

**Quotient audit**
- T3, where the quotient is violated: p < 0.05 in 3/12 runs.
- T1, where it is valid: 1/12.

The audit has low power at this size. In T3 the bounds still stayed valid, because the extra mixing from the bad quotient raised the representation floor less than the certificate's slack.

**Tipping point.** The pair route stays positive up to a declared δ ≈ 0.49–0.5. The accepted route reaches 0 at δ ≈ 0.06 for K = 2 and ≈ 0.11 for K = 4.

## 2. Real labels: MTSamples specialty from exact duplicate text

20 development/certification splits by cell.

| Labels | Plug-in | Accepted bound | Pair (unaudited) | Linear (unaudited) | Within-cell agreement | Independence audit | **Pythia after audit** |
|---|---|---|---|---|---|---|---|
| 40 specialties | 0.526 | 0.486 | 0.651 | 0.496 | 0.001 (iid would give ≥ 0.025) | rejects, p ≈ 0 | **0.486** (refuses U-statistic routes) |
| Top 10 + other | 0.497 | 0.460 | 0.637 | 0.867 | 0.043 (iid would give ≥ 0.091) | rejects, p ≈ 0 | **0.460** |

**Finding.** Duplicate notes are listed under different specialties by construction: 2 same-label pairs out of 3,105. That is sampling without replacement, not repeated draws. The pair and minorant routes would overstate the floor here, and the audit refuses them. The accepted min-count route needs only each row's marginal law, so it stays valid.

## 3. Real tabular covariates: PMLB pharynx (1M rows as the population), planted labels

20k certification rows, 6 runs. Representation ladder:
- **S1:** T, N
- **S2:** + Site, Grade
- **S3:** + Condition, Inst
- **S4:** + sex, Treatment, Age in 5 bins
- **S5:** Age in 20 bins

| K | Representation | Median cell rows | Singleton share | Accepted | Pair | Linear | **Pythia** |
|---|---|---|---|---|---|---|---|
| 2 | S1 | 1882 | 0 | **0.98** | 0.85 | 0.90 | **0.98** |
| 2 | S3 | 22 | 0.04 | **0.87** | 0.73 | 0.82 | **0.87** |
| 2 | S4 | 2 | 0.37 | 0.46 | 0.45 | **0.50** | **0.50** |
| 2 | S5 | 1 | 0.68 | 0.22 | 0.23 | **0.24** | **0.24** |
| 3 | S1 | 1902 | 0 | **0.97** | 0.83 | 0.88 | **0.97** |
| 3 | S4 | 2 | 0.37 | **0.41** | 0.39 | 0.40 | **0.41** |
| 3 | S5 | 1 | 0.68 | 0.18 | 0.18 | **0.18** | **0.18** |

- Violations: 0/60.
- With large cells, the existing row-level domination route is already 0.86–0.98 of the floor, and the new routes add nothing.
- In fragmented representations they add a little (+0.03 for K = 2). The bound there is capped by rows that have no replicate at all.

**Real labels on the same ladder (no gold).** The approximate reference is the plug-in floor on 400k rows.

| Target | S1 | S3 | S4 | S5 |
|---|---|---|---|---|
| Status, Pythia vs accepted | 0.262 vs 0.262 (ref 0.269) | 0.222 vs 0.222 (ref 0.250) | **0.130** vs 0.112 (ref 0.221) | **0.059** vs 0.049 (ref 0.179) |
| Grade, Pythia vs accepted | 0.427 vs 0.427 (ref 0.435) | 0.413 vs 0.413 (ref 0.436) | **0.328** vs 0.287 (ref 0.417) | **0.206** vs 0.160 (ref 0.376) |

**Second-order slack on tabular data (certify S5 through S4 cells).** Pair via S4 gives 0.04–0.08 for δ = 0.05–0.3. Min-count gives 0 at every δ. Direct S5 gives 0.03–0.04.
- Here the coarser S4 is itself fragmented, so the gain is small.
- The true maximum within-cell TV is large (0.33 for K = 2), though the median is small (0.06).

## Reading

1. **Raw clinical text.** Exact duplicates give almost nothing (0.04). A declared nuisance quotient turns the same notes into usable replicates (0.58–0.66 of the floor), with no embedding, no metric and no smoothness constant. This is the embedding-free mechanism that the metric routes lacked.
2. **Copy-forward / near-duplicates.** The second-order slack is the decisive new result. At the same declared tolerance, the accepted route collapses (0.04 or 0.26), while Pythia keeps 0.55–0.57.
3. **Small cells.** The unbiased routes beat plug-in optimism when cells hold 2–5 rows. When cells are large, the existing domination route is already near-tight, so the portfolio keeps both.
4. **New failure mode found and guarded.** U-statistic routes require conditionally independent replicates. Real MTSamples duplicates violate this, the independence audit catches it, and Pythia falls back to the min-count route.
5. **Remaining cap.** Rows without any replicate. The two levers are a better quotient (more nuisance declared and audited) or near-replicates with the δ² slack.
6. **kNN.** The 1-NN TF-IDF Cover–Hart value (0.81–0.88) is an asymptotic estimate. It needs a metric and carries no finite-sample guarantee, so it is not comparable as a certificate. It also stayed below the floor here, which shows how much room remains between certified and estimated values.
