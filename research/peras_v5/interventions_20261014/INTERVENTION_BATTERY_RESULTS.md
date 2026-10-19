# Finite-text intervention battery results

Protocol: `INTERVENTION_BATTERY_REGISTRATION.md`.
Raw rows: `intervention_battery_results.jsonl`.
SHA-256: `d538d20f2eaea40e499f700811eaa6e2246bde948ca80a9e3de33987e8c72e57`.

The frozen run contains 5,000 rows: ten known-law mechanisms, 50 untouched
seeds and ten representation interventions. The public finite-cell route
accepted 4,000 outcome-blind rows and refused all 1,000 target-derived rows.
There were zero accepted lower-bound exceedances of the exact Bayes MAE.

## Aggregate results

| Representation | Accepted | Positive bounds | Median bound / gold | Median external-estimator relative error | Median cells | Median singleton fraction |
|---|---:|---:|---:|---:|---:|---:|
| baseline cell text | 500 | 450 | 0.525 | 0.021 | 20 | 0.000 |
| bijective cipher | 500 | 450 | 0.525 | 0.021 | 20 | 0.000 |
| nuisance ignored canonically | 500 | 450 | 0.525 | 0.021 | 20 | 0.000 |
| nuisance retained naively | 500 | 218 | **0.000** | **0.394** | 706 | 0.452 |
| five-family garbling | 500 | 500 | 0.602 | 0.015 | 5 | 0.000 |
| within-family reassignment | 500 | 500 | 0.592 | 0.020 | 20 | 0.000 |
| global note misalignment | 500 | 500 | 0.686 | 0.022 | 20 | 0.000 |
| token shuffle, bag parser | 500 | 450 | 0.525 | 0.021 | 20 | 0.000 |
| literal target sentence | **0** | 0 | — | diagnostic 0 | — | — |
| unary `X` target code | **0** | 0 | — | diagnostic 0 | — | — |

The larger ratios after garbling or misalignment do not mean those
representations are more informative. Their true Bayes floors are larger, and
coarser cells provide more repeat observations per action. Tightness is a
property of the witness relative to its own estimand, not an information-value
ranking.

## Metamorphic checks

Every paired check below held in 500/500 cases:

- bijective cipher: exact gold and numerical bound equality;
- canonical nuisance parser: exact gold and numerical bound equality;
- bag-of-token shuffle: exact gold and numerical bound equality;
- naive nuisance retention: gold equality and bound no larger than baseline;
- deterministic family garbling: gold no smaller than baseline;
- within-family reassignment: exact equality to the family-only gold;
- global misalignment: gold no smaller than family-only gold.

This separates two phenomena that earlier Mayo perturbations mixed together:

1. **population information**, governed by the registered intervention
   contract; and
2. **witness resolution**, governed by cell/neighborhood geometry.

Adding an independent identifier cannot change true Bayes risk, but it split
the median certification cohort into roughly 706 cells, 45% of entities in
singletons. The lower endpoint remained valid while its median tightness fell
from 0.525 to zero. The external split cell-median estimate also degraded from
2.1% to 39.4% median relative error. A canonical parser that discards the
nuisance restored both results exactly.

## Known weakness

The high-SNR mechanism remained zero-bound in all 50 baseline seeds: true MAE
0.120, median bound 0. This is concentration width, not transfer penalty. The
finite-cell route uses the global hard outcome range in McDiarmid; high signal
across cells enlarges that cap even though within-cell noise is small. A
cell-stratified or variance-adaptive concentration theorem is the next bound
target.

The target sentence and unary `X` code deliberately expose the target. Their
diagnostic Bayes risk is zero, but the public route refuses them because cell
labels are not outcome-blind. This is the required behavior, not a missed
certificate.
