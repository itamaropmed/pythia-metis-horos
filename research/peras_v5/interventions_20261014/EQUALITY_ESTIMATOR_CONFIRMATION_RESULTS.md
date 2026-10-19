# Equality-class estimator confirmation results

Date: 2026-10-14

The preregistered least-fragmented equality router was tested on seeds
`64001`--`64050`, which were not used by the preceding bound or lattice
development.

## Primary results

- Paired absolute-error wins: **496 / 500**; no ties.
- Median absolute error of raw nuisance-fragmented SAA: **0.134457**.
- Median absolute error after routing to the canonical equal-risk
  neighborhood: **0.013032**.
- Mechanisms with improved median absolute error: **10 / 10**; a two-sided
  mechanism-level sign test gives `p = 0.001953125`.
- Correct canonical routing source: **500 / 500**.
- Raw and lattice-closed known-risk exceedances: **0 / 2,500** each.
- Order failures and equality-repair failures: **0 / 500** each.

| Mechanism | Raw nuisance median absolute error | Routed median absolute error | Paired wins / 50 |
|---|---:|---:|---:|
| bimodal | 0.337431 | **0.026535** | 50 |
| discrete mixture | 0.071777 | **0.014160** | 47 |
| Gaussian | 0.189536 | **0.016235** | 50 |
| heteroscedastic | 0.197315 | **0.016540** | 50 |
| high SNR | 0.028430 | **0.002435** | 50 |
| lognormal | 0.103161 | **0.013506** | 49 |
| low SNR | 0.379071 | **0.032471** | 50 |
| t3 | 0.134865 | **0.014442** | 50 |
| triangular | 0.080733 | **0.006384** | 50 |
| uniform | 0.134107 | **0.011613** | 50 |

## Interpretation

The theorem supports a one-sided bias statement: within a genuinely
equal-risk class, empirical refinement can only decrease the in-sample SAA
optimum, so the least-fragmented plug-in has weakly smaller downward bias in
expectation.  The observed absolute-error improvement is empirical and was
confirmed on fresh mechanisms/seeds; it is not a theorem of universal MSE or
absolute-error dominance.

The rule uses only frozen labels and registry order.  It does not inspect
outcomes to select a representation and does not train a model.  Its validity
depends critically on the external proof that the nuisance representation is
information-equivalent to the canonical representation.

Raw artifact:
`equality_estimator_confirmation_results.jsonl`

SHA-256:
`6b030646aa42786cc8a79cb0794f7a7028b846b2ffd8767ba82ace1485755dd6`
