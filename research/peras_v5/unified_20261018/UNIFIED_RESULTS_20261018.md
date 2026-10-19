# PYTHIA-U: one simultaneous certificate for table + text (2026-10-18)

- **Theory:** `docs/theory/PYTHIA_U_UNIFIED_CERTIFICATE_20261018.md`.
- **Code:** `pythia/core/unified.py`.
- **Run scripts:** `run_unified.py` (generated), `run_unified_real.py` (MTSamples), `summ_unified.py`.
- **Logs:** `log_port_*.txt` (main: LR + gradient-boosting portfolio), `log_gen_*.txt` (logistic-regression-only ablation), `log_real.txt`.

## What one certificate now says

All statements hold simultaneously at α = 0.05, on the certified patients. A population version is also given.

| Statement | Side | Assumption |
|---|---|---|
| Best achievable risk using the **table**: G + V ∈ [t_L, t_U] | two-sided | none |
| **Table information** relative to no information | two-sided | none |
| Best achievable risk using **table + notes**: G ≤ U | ceiling | none |
| | floor ≥ L | A(½) with slack ε, plus the tipping slack ε\* |
| **Text information beyond the table** (NLI): V ≥ V_L | lower | none |
| Model headroom: R(f) − G ≥ R_L(f) − U | lower | none |
| Falsification of A(½) when NLI and the pair designs conflict | test | none |
| Loss transfers: Brier, 0-1, log, conditional MI | — | — |

## 1. Generated data with exact laws

n = 8000, 10 datasets × 5 replicates = 50 certificates. In each certificate, every bound is checked against the exact truth on the certified rows. Ratios are means.

| Dataset | True G | PYTHIA-U G interval | U / true | Earlier LP-1 U / true | True NLI V | **PYTHIA-U V ≥** | Earlier LP-1 V ≥ | Earlier AFI V ≥ | Binding ceiling route |
|---|---|---|---|---|---|---|---|---|---|
| D01 echo (V = 0) | 0.420 | [0.042, 0.440] | 1.05 | 1.04 | 0 | 0 | 0 | 0 | pair |
| D02 novel findings | 0.392 | [0.122, 0.425] | 1.09 | 1.08 | 0.079 | **0.045** | 0.017 | 0.026 | pair |
| D04 synergy | 0.372 | [0.174, 0.418] | 1.12 | 1.12 | 0.093 | **0.042** | 0.021 | 0.038 | pair |
| D08 outcome leak | 0.293 | [0.008, **0.365**] | **1.25** | 1.41 | 0.126 | **0.069** | 0 | 0 | model |
| D09 4-class | 0.414 | [0.218, 0.462] | 1.12 | 1.11 | 0.075 | **0.036** | 0 | 0.013 | pair |
| O1 graded severity | 0.398 | [0.001, 0.444] | 1.12 | 1.12 | 0.054 | **0.020** | 0 | 0 | model / pair |
| O2 long-range negation | 0.399 | [0.030, 0.439] | 1.10 | 1.10 | 0.052 | **0.022** | 0 | 0.001 | pair / model |
| O4 copy-forward | 0.400 | [0.171, **0.440**] | 1.10 | 1.12 | 0.050 | **0.026** | 0 | 0 | model |
| O5 two languages | 0.398 | [0.014, 0.454] | 1.14 | 1.15 | 0.053 | **0.014** | 0 | 0 | model / pair |
| D10 style confound | 0.402 | [0.118, 0.424] | 1.06 | 1.05 | 0.019 | 0.003 | 0 | 0 | pair |

**Validity**
- 0 violations in 50 certificates. The checks covered both sides of G, V, the table floor, the population interval and the stratified intervals.
- A(½) was never falsely refuted (0 of 50).

**Text information beyond the table**
- Certified share of the truth: **42%** on average (logistic regression alone: 35%), against 5% for LP-1 and 11% for AFI. That is a 4–8× gain.
- In 6 of 9 signal datasets the earlier methods certified nothing at all.

**Joint ceiling**
- Unchanged where text neighbours are informative: 1.05–1.14 × the truth. The new parts cost about 0.002 there, because they take part of α.
- Clearly tighter where the held-out model beats neighbour pairs: outcome leak 1.41 → **1.25**, copy-forward 1.12 → 1.10.

**Model portfolio**
- Adding a gradient-boosting model to logistic regression raised the synergy dataset from 0.009 to 0.042. Logistic regression cannot represent the interaction.
- Validity never depends on the models. Each model only spends its share of α.

**Table floor**
- Two-sided interval width 0.078. The interval contained the truth every time.
- Table information is certified as [0.009, 0.145] on average.

**Table-conditional form (strata = age band)**
- It is valid, but on these data it is looser than the pooled certificate: ceiling 1.20 × truth, and 14% of V. The cause is α split across strata and smaller samples.
- Use it as a *map* of where the notes add information, not to tighten the global bound.

## 2. Real MTSamples (2,357 notes; mean of 3 seeds)

| Outcome | Table | Joint ceiling U | **Text info V ≥ (PYTHIA-U)** | LP-1 V ≥ | AFI V ≥ | Table floor [L, U] | Table info [L, U] |
|---|---|---|---|---|---|---|---|
| Specialty (K = 11) | note type | 0.268 | 0.261 | **0.281** | 0.276 | [0.53, 0.67] | [0.03, 0.32] |
| | note type × age × sex | 0.276 | **0.208** | 0.104 | 0.143 | [0.48, 0.63] | [0.04, 0.35] |
| Surgery vs not | note type | 0.068 | 0.119 | **0.136** | 0.137 | [0.18, 0.33] | [0.08, 0.38] |
| | note type × age × sex | 0.072 | **0.114** | 0.065 | 0.074 | [0.18, 0.33] | [0.08, 0.39] |

**With the richer table, NLI doubles the certified text information** (0.21 vs 0.10–0.14, and 0.11 vs 0.07). The pair route weakens as cells get small; the NLI route does not.

**With the coarse table, the earlier pair routes are slightly ahead** (by 0.02). The unified certificate spends part of α on its extra components.

**Conditional map, surgery vs not (stratified certificate)**
- Within operative notes: text information ≥ 0.216, joint ceiling 0.104.
- Every other note type: text information 0, because those strata contain essentially no surgery cases.

## Reading

1. **One certificate, all quantities.** The table floor, table information, joint ceiling, text information and assumption budget now come from one LP over one estimand, valid together (0 of 50 violations).
2. **NLI is where most of the gain is.**
   - Text information beyond the table is certified 4–8× more strongly on generated data, and 1.5–2× on real data with a rich table.
   - The joint ceiling improves only where a model beats neighbour pairs.
3. **Still loose**
   - The joint *floor* (the lower side) still needs A(½) and is far from the truth.
   - The table box costs ±0.04 at n = 8000, because the bounded-difference bound is conservative.
   - Stratified certificates pay for multiplicity.

**Next**
- A variance-adaptive table box.
- Folding the AFI anchored ceiling in as one more LP constraint. It is 0.5–3% tighter than the pair boxes on some datasets.
- A credentialed MIMIC run, with aggregates only.
