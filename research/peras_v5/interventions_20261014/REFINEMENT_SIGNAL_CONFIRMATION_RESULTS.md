# Finite-refinement signal and null-calibration results

Date: 2026-10-14

## Power battery

The registered SAA-drop statistic detected the known pre-outcome concept in
**500/500** runs at the two-family threshold 0.025, including 50/50 in every
one of the ten mechanisms.  Median observed empirical MAE-floor reduction was
0.426797.  No predictor was trained.

The independent-nuisance control rejected in 29/500 mechanism rows.  Those
rows were correlated because each integer seed was reused across ten noise
mechanisms: the 29 rejections clustered in four seed IDs.  This satisfies the
loose preregistered ceiling of 40 but is not treated as a 500-independent-run
calibration result.

Raw power artifact SHA-256:
`1b7c19cd932fbbdc4e588eb4b0ba10fbece313c84ec1e0c24eaa59b427877036`

## Independent null audit

A separately registered audit used 1,000 disjoint seeds, 199 permutations and
2-, 4- and 8-state nuisance refinements.

- Rejections at nominal 0.05: **56/1,000 = 5.6%**.
- Two-sided 95% Wilson interval: **4.337%--7.202%**.
- The interval contains 5%, and its upper endpoint is below the registered 8%
  gate.
- Mechanism rejection counts ranged from 4/100 to 7/100.
- Nuisance-cardinality counts were 19/340, 13/330 and 24/330 for 2, 4 and 8
  states.

Raw null artifact SHA-256:
`969d1d0d738fe720f033d8841c4c4dff633698404306917cbde07ee5fbae5506`

## Claim boundary

Under exact within-coarse exchangeability and outcome-blind frozen labels, the
plus-one permutation p-value is finite-sample valid.  The statistic tests
`Y independent of refinement conditional on coarse label`; it does not lower-
bound the magnitude of population information value.  When exchangeability is
not independently attested, the public API labels its number an exploratory
Monte Carlo tail area rather than a valid p-value.
