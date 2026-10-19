# LP bounds, relaxation paths and text-geometry maps (development, 2026-10-17)

Theory: `docs/theory/LP_RELAXATION_AND_TEXT_GEOMETRY_20261017.md`. Brier units are 1 − |p|².

## 1. Generated data with exact truth (n = 8000, 5 runs each)

Each cell is the mean over runs.

| | D01 echo | D02 novel | D04 synergy | D09 4-class | Regression (MSE, estimation task) |
|---|---|---|---|---|---|
| True joint floor G | 0.421 | 0.392 | 0.373 | 0.415 | 0.726 |
| LP floor, γ = 1 | 0.259 | 0.277 | 0.326 | 0.399 (**2/5 violations**) | 0.119 |
| **LP floor, γ = ½** | 0.086 | 0.141 | 0.159 | 0.198 | 0 |
| LP ceiling, assumption-free | 0.442 | 0.434 | 0.416 | 0.465 | 1.018 |
| True text info V | 0 | 0.079 | 0.092 | 0.074 | 0.548 |
| V lower bound (assumption-free) | 0 | 0.009 | 0.022 | 0.001 | 0 |
| **Tipping slack ε\*, γ = ½** (assumption budget) | 0.015 | 0.029 | 0.028 | 0.034 | 0 |
| Tipping slack ε\*, γ = 1 | 0.091 | 0.112 | 0.104 | 0.126 | 0.048 |
| True table 0-1 floor | 0.334 | 0.401 | 0.390 | — | — |
| Moment LP lower, J = 2 (Jensen-level) | 0.218 | 0.267 | 0.261 | — | — |
| **Moment LP lower, J = 3** | 0.232 | 0.288 | 0.280 | — | — |
| Moment LP upper, J = 3 | 0.483 | 0.444 | 0.499 | — | — |
| Map: Spearman of per-cell ΔI estimate with true per-cell V | 0.01 (truth all 0) | 0.29 | 0.36 | 0.68 | — |
| Map: cells certified / false certifications | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | — |

**Validity**
- Every assumption-free bound and every γ = ½ bound held in all 25 runs.
- γ = 1 broke on the 4-class dataset in 2 of 5 runs. This is the known risk of γ = 1, now visible as a violation.

**Relaxation path**
- The floor falls smoothly as the slack ε grows. With γ = ½ the certificate tolerates shape violations of about 0.015–0.034 in Brier units before it reaches 0. That is a concrete, reportable *assumption budget*.

**Moment LP**
- The third moment (agreement among triples) lifts the table 0-1 floor by about 0.015–0.02 over the Jensen level. It is still looser than fixed-action domination, which reached about 0.37 on similar data.
- The moment LP's value is its sharp two-sided interval and its ability to absorb text-group drift.

**Maps**
- The ordering of cells is informative: Spearman 0.3–0.7 with the true per-cell text information.
- At about 330 rows per cell, no single cell survives the Holm correction.

## 2. Real MTSamples (2,357 notes)

| Outcome | Table | Joint floor ceiling (no assumption) | LP floor, γ = ½ | Certified text info beyond table |
|---|---|---|---|---|
| Specialty (K = 11) | note type | 0.273 | 0 | **0.275** |
| | note type + age + sex | 0.271 | 0 | 0.042 |
| Surgery vs not | note type | 0.064 | 0 | **0.140** |
| | note type + age + sex | 0.069 | 0 | 0.008 |

On real notes the text variogram rises steeply, so the γ-floor is 0. This matches Theorem 4: from text, only the ceiling and the information lower bound are robust.

**Conditional text-geometry map (specialty, table = note type; Holm-certified lower bounds)**

| Note type | Rows | Cell disagreement | Text info ΔI (point) | **Certified ΔI ≥** | Share resolved by text |
|---|---|---|---|---|---|
| Operative | 1,367 | 0.485 | 0.333 | **0.217** | **69%** |
| Discharge | 218 | 0.812 | 0.353 | 0.093 | 44% |
| H&P / consult | 282 | 0.851 | 0.375 | 0.134 | 44% |
| Imaging / report | 235 | 0.543 | 0.312 | 0.085 | 58% |
| Other | 255 | 0.813 | 0.436 | 0.172 | 54% |

**Surgery vs not.** The notes carry signal only inside operative notes, with certified ΔI ≥ 0.234 and 84% resolved. Discharge, H&P and imaging cells have essentially no outcome disagreement, so there is nothing for the text to add there.

## Reading

1. **The LP unifies every certificate.** A single program combines all of them:
   - the pair identity, the table cells, the exact drift ≤ 2V fact and the γ shape;
   - two design families;
   - an explicit relaxation dial with a tipping point.
2. **The relaxation path is what industry should report.** For example: "the floor is 0.14 if the outcome law's roughness respects γ = ½; it survives violations up to ε\* = 0.03; the ceiling and text-information bounds need no assumption".
3. **The conditional geometry map** is a new, practical diagnostic. It certifies *which table strata* have notes that carry signal beyond the structured fields.
