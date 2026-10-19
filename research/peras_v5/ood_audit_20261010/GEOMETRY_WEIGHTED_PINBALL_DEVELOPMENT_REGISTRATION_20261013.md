# Geometry-weighted Q90 mixture development registry

Written before seeds `138001--138100`.

- Ten bounded near-replicate scale-law mechanisms with analytic average Q90
  risk and analytic partition-specific comonotone barycentric discrepancies.
- 800 independent units with fixed scalar feature `x`.
- 20 outcome-blind partitions of size 16, produced by sorting increasingly
  jittered versions of `x`.
- Residual clip `[0,2]`, Q90 score cap `.16`, alpha `.05`, one candidate per
  route.
- Comparators: single minimum-declared-penalty partition and equal e-mixture.
- Development weight families:
  `exp(-(P_r-min P)/(tau*median positive P))` for
  `tau=(.05,.1,.25,.5,1)`.
- Outcomes never enter partition weights or discrepancy declarations.

This is development only.  At most one weight rule may be frozen for new
seeds.  Success requires validity and a mechanism-level advantage over the
single best predeclared geometry partition, not merely over paired Bentkus.
