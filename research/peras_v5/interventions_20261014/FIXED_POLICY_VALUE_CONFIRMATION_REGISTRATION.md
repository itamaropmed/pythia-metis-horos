# Registered fixed-policy information-value confirmation

Written after the 2,048-entity resolution study and before opening seeds
`68001`--`68050`.

The 2,048-entity development run was valid but could not resolve the
approximately 0.065-unit uniform-noise information value.  The confirmation
therefore freezes 8,192 entities, comparable to the order of the Mayo unique-
patient cohort, before opening new seeds.

## Frozen design

- Ten bounded residual mechanisms and 50 fresh seeds per mechanism.
- Coarse representation: procedure.
- Refined representation: procedure plus a binary pre-outcome concept.
- Externally declared policy: analytic conditional median using the refined
  synthetic information; Peras receives only its losses and does no fitting.
- Coarse endpoint: cell-local finite SAA.
- Policy-risk upper endpoint: predeclared dKelly route.  Earlier independent
  calibration already registered dKelly for absolute discrepancy; it is not
  selected from confirmation outcomes.
- Family alpha 0.05, split equally between coarse lower and policy upper.
- Matched baseline: global-range coarse SAA plus Hoeffding policy upper at the
  same family alpha.

## Primary endpoints

1. zero information-value lower endpoints above analytic gold;
2. positive value endpoint in all 500 runs and every mechanism;
3. strict improvement over the matched baseline in a majority of runs and a
   majority of mechanism medians;
4. median bound/gold above 0.60;
5. positive endpoint in all 50 uniform-noise runs.

This confirmation evaluates the bridge when a genuinely frozen refined policy
exists.  It does not license using retrospectively selected or ordinary OOF
predictions as if they were an independent holdout policy.
