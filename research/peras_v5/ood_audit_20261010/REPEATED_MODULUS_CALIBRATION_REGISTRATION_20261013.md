# Repeated-environment modulus calibration: frozen audit

This registration was written before opening seeds `133001--133100`.

- Ten affine-quantile mechanisms defined in
  `run_repeated_modulus_calibration_ood.py`.
- 80 fixed units and 20 globally disjoint blocks of four.
- 500 independent common-uniform environment rows per run.
- Every environment coordinate has its target unit-conditional uniform law;
  within-row dependence is deliberately perfect and allowed.
- Fixed clip `[-5,5]` and Euclidean `[x,note]` feature metric.
- Family alpha `.05`, divided across absolute/squared penalty families and
  Hoeffding, Maurer--Pontil empirical Bernstein, and Bentkus endpoints.
- Primary validity endpoints: zero misses of the analytic absolute and squared
  barycentric population costs among 1,000 runs each.
- Secondary endpoints: median cap/truth ratio and selected-method counts.
- The same resource also permits direct repeated-environment SAA; this audit
  tests the calibration theorem rather than claiming resource-matched bound
  superiority.
