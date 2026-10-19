# NLI (Natural Language Information): unseen-data tests against literature methods (2026-10-17)

- **Theory:** `docs/theory/NATURAL_LANGUAGE_INFORMATION_NLI_20261017.md`.
- **Code:** `methods.py`, `generator_ood.py`, `run_nli.py`, `run_nli2.py`, `run_nli_real.py`, `run_nli_scale.py`, `summarize_nli.py`.
- **Full outputs:** `summary_ood.txt`, `summary_real.txt`, `log_*.txt`.
- **Tests:** `tests/test_nli.py`.

## Design

**Six new families (never used during development).** Each has a known outcome law, so the true NLI is exact. At strength 0, Y ⊥ T | X exactly, yet the note still echoes the table and carries cell-tied writing style.

| Family | What the note carries |
|---|---|
| O1 | Severity as graded synonyms |
| O2 | Negation far from the finding (word order kept) |
| O3 | Lab values spelled out in words |
| O4 | Copy-forward templates plus a 2-token new finding |
| O5 | Two vocabularies ("languages") with a language-specific cue |
| O6 | A rare token that matters only in one age band |

**Runs:** n = 4000, 24 table cells, 25 replicates × 3 strengths (true NLI ≈ 0, 0.015, 0.05 Brier units). That is 450 datasets. All methods saw exactly the same data.

**Real notes.** MTSamples (2,357 real transcriptions); the table is note type, or note type × age × sex. Three settings:
- the real outcome, surgery vs not;
- an exact null, with labels permuted within table cells;
- a semi-synthetic signal: real notes, with the outcome driven by whether the note mentions "hypertension". This keeps real clinical language and known truth.

25 replicates × 2 tables for each.

**Methods** (α = 0.05; all model-based methods use the same logistic regression on table one-hot + TF-IDF)

| | Method | What it is |
|---|---|---|
| **Ours** | NLI-proj-perm | Theorem 6: exact within-cell permutation of Σ yᵢhᵢ |
| | NLI-proj-cert | Theorem 5: certified lower bound on NLI. Reported as a population bound and as a cohort bound |
| | NLI pair route | Earlier model-free AFI geometry certificate and pair permutation test |
| **Baselines** | Boot-ΔBrier | Cross-fitted ΔBrier with a percentile bootstrap: common practice, no guarantee |
| | LocalPerm-ΔBrier | Kim et al. 2022 style local permutation |
| | CODEC-perm | Azadkia–Chatterjee coefficient, within-cell permutation |
| | WGCM | Weighted generalized covariance measure, normal approximation |

## 1. Validity: no false claims anywhere

| | Rejections under the exact null (150 OOD + 50 real) | Certified bound above the truth |
|---|---|---|
| NLI-proj-perm | 4.7% [2.3, 9.3] OOD; 4.0% real | — |
| NLI-proj-cert (population) | 0 / 200 | **0 / 600 runs** |
| NLI-proj-cert (cohort) | 1.3% OOD, 0% real | **0 / 400** signal runs |
| Pair certificate | 0 | 0 / 600 |
| LocalPerm / CODEC / WGCM | 4.0% / 2.7% / 3.3% OOD | — |
| Boot-ΔBrier | 0% | — |

Boot-ΔBrier never rejects under the null, but it is conservative for a bad reason: the model overfits and ΔBrier turns negative.

## 2. Detection power on unseen families (levels 1 and 2 pooled, n = 300 datasets)

| Method | Power | Ours-only / baseline-only rejections | Exact McNemar p |
|---|---|---|---|
| **NLI-proj-perm** | **0.84** | — | — |
| LocalPerm-ΔBrier | 0.83 | 8 / 5 | 0.58 (tie) |
| WGCM | 0.84 | 3 / 1 | 0.62 (tie) |
| Boot-ΔBrier (common practice) | 0.50 | **104 / 0** | **1e-31** |
| CODEC-perm | 0.33 | **160 / 5** | **4e-41** |
| NLI-proj-cert > 0 (a *certified magnitude*, not just a p-value) | 0.61 | vs Boot: 36 / 3 | **4e-8** |
| Earlier NLI pair certificate | 0.007 | — | — |

**Weakest signal (true NLI ≈ 0.016):**
- NLI-proj-perm: 0.80
- LocalPerm: 0.79
- Boot-ΔBrier: 0.16
- CODEC: 0.14

**Per family.** NLI-proj-perm ties the best valid baseline in every family (all p = 1). O6 (rare local token) defeats every method (≈ 0.3).

## 3. Real notes (MTSamples)

| | NLI-proj-perm | LocalPerm | WGCM | Boot-ΔBrier | CODEC | NLI-proj-cert > 0, population / cohort |
|---|---|---|---|---|---|---|
| Null (size) | 0.04 | 0.06 | 0.04 | 0 | 0.06 | 0 / 0 |
| Semi-synthetic, NLI ≈ 0.015 | 0.52 | 0.50 | 0.58 | **0.00** | 0.10 | 0.02 / 0.14 |
| Semi-synthetic, NLI ≈ 0.035 | **0.98** | 0.96 | 0.98 | 0.18 | 0.12 | 0.42 / **0.78** |

Against Boot-ΔBrier, NLI-proj-perm won on 66 discordant datasets and lost on 0 (p = 3e-20). Against CODEC it won 65 to 1 (p = 2e-18).

**Real outcome (surgery vs not)**

| Table | Certified NLI ≥ | Notes |
|---|---|---|
| Note type | **0.094** (projection), 0.122 (pair route) | Model ΔBrier point estimate 0.153 |
| Note type × age × sex | **0.085** | |

All tests give p ≤ 0.01.

## 4. Magnitude: the certificate is valid but conservative at n = 4000, and it converges

**At n = 4000, level 2 (true NLI ≈ 0.05)**
- Population certificate: 8% of the truth.
- Cohort certificate: 21% of the truth.
- Boot-ΔBrier lower limit: 25%, with no guarantee. At weak signal its estimate is negative in 51–94% of runs.

**Scaling, 5 replicates per size, 0 violations**

| n | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| O1: population certificate / truth | 0.01 | 0.05 | 0.18 | 0.36 | **0.54** |
| O1: cohort certificate / truth | 0.11 | 0.17 | 0.33 | 0.50 | **0.64** |
| O1: model correlation / ideal √(NLI/2) | 0.55 | 0.59 | 0.70 | 0.80 | 0.87 |
| O4 (NLI 0.015): cohort certificate / truth | 0 | 0.03 | 0.14 | 0.23 | 0.37 |

**Where the gap comes from.** Most of it is the *model* (ρ/ideal < 1), not the inequality. The certificate equals 2(ρ − O(n^{-1/2}))², so any better text model tightens it automatically, and validity never depends on the model.

## Verdict: is there a significant win?

**1. Detection: no win over the strongest valid tests.**
- NLI-proj-perm ties LocalPerm-ΔBrier and WGCM, both overall and in every family.
- Theorem 6's remark predicts this: within cells, the permutation laws of these statistics coincide. This is reported honestly, not as a defeat.
- No valid test can do much better here without a better text model.

**2. Highly significant wins over common practice and over CODEC.**
- Against the bootstrap ΔBrier that applied papers actually use: +34 points of power, McNemar p ≈ 1e-31, and 0.98 vs 0.18 on real notes.
- Against CODEC: McNemar p ≈ 4e-41.

**3. The real, new capability is a certified amount of information.**
- Only NLI turns any model into a *guaranteed* lower bound on how much the notes add beyond the table, valid at the null boundary: 0 of 600 violations.
- That bound is positive in 61% of unseen signal datasets (78% on real notes at NLI ≈ 0.035). The earlier model-free pair route managed 0.7%.
- No accepted method returns a valid magnitude. LocalPerm, WGCM and CODEC give p-values only. The bootstrap has no guarantee, and its CI fails at the boundary.

**4. Weak points**
- The magnitude is conservative at n ≤ 4k: 8–21% of the truth.
- O6-type signals (rare, local) remain hard for every method.
- The upper side is impossible without assumptions (Theorem 7).
