# Registered repeated-action-dispersion confirmation

Written before generating the confirmation rows.

## Protocol

- Ten fixed-unit action/scale geometries: linear, nonlinear, sparse-step,
  high-frequency, heteroscedastic, discrete-cell, interaction, skewed-action,
  high-SNR and low-SNR.
- Sixteen fixed units in two disjoint blocks of eight.
- 800 independent repeated environments per run; arbitrary dependence across
  units within a row is induced by a shared uniform shock plus independent
  uniform noise.
- Quantile actions: median/MAE (`q=.5`) and Q90 (`q=.9`).
- Exact analytic hard support and analytic quantile actions.
- Simultaneous interval alpha: 0.01.
- Untouched seeds 51001--51050: 1,000 rows total.

## Endpoints

For each row, compare the returned caps with the exact block-average action
variance, median absolute action dispersion, maximum arithmetic-centre
deviation, and optimized curvature-weighted quadratic penalty.  Report misses,
cap/truth ratios on positive truths, and a Wilson lower confidence limit for
the all-metric row coverage.  The unit of repetition is a complete mechanism,
seed, and quantile row.

This battery tests the non-oracle calibration bridge.  It does not re-test the
downstream score concentration and does not change Mayo.

