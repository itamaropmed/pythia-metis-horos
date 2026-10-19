# Repeated-action-dispersion confirmation results

Protocol: `ACTION_DISPERSION_CONFIRMATION_REGISTRATION_20261014.md`.
Raw rows: `action_dispersion_confirmation_results.jsonl`.
SHA-256: `870d3275a5b34e160cd1c6cb6df858bebb001326e8ce3a66f1adc0b41ca2c431`.

The frozen battery contains 1,000 rows: ten mechanisms, 50 untouched seeds,
and median/Q90 actions.  Every row used 800 independent environments with a
shared within-row shock, two disjoint blocks of eight, and simultaneous
order-statistic alpha 0.01.

## Validity and tightness

All four exact targets were covered in 1,000/1,000 rows: action variance,
median absolute action dispersion, maximum action displacement, and optimized
curvature-weighted quadratic penalty.  The 95% Wilson lower limit for joint
row coverage is 0.9962.

| Quantile action | Target | Median cap / truth | Worst cap / truth | Misses |
|---|---|---:|---:|---:|
| median / MAE | variance | 1.217 | 6.131 | 0/500 |
| median / MAE | MAD | 1.160 | 2.758 | 0/500 |
| median / MAE | max displacement | 1.151 | 2.936 | 0/500 |
| median / MAE | curvature penalty | 1.221 | 6.131 | 0/500 |
| Q90 | variance | 1.279 | 8.553 | 0/500 |
| Q90 | MAD | 1.194 | 3.262 | 0/500 |
| Q90 | max displacement | 1.196 | 3.428 | 0/500 |
| Q90 | curvature penalty | 1.284 | 8.553 | 0/500 |

The median caps are practically useful.  The worst variance/curvature ratios
show the expected weak-signal failure: when true between-action dispersion is
small, simultaneous unitwise intervals can be several times the target.  A
future block-level confidence region may tighten this, but it must preserve
coverage on a fresh registry.

## Claim

This confirms the non-oracle repeated-environment bridge under its declared
contract.  It does not make the current Mayo study eligible: Mayo has one
duration outcome per case rather than repeated environments of the same fixed
case.  For Mayo, the curvature route remains a declared sensitivity curve
unless an external action-dispersion resource is obtained.

