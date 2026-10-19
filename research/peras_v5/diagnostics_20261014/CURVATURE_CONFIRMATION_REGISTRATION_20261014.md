# Registered curvature-penalty confirmation

This file was written before the confirmation output was generated.

## Question

Does the declared second-order Bayes-action-drift correction preserve validity
and improve tightness over the existing first-order barycentric correction when
all concentration choices and score caps are frozen independently of the
certification outcomes?

This is an **oracle-declaration diagnostic**, not a deployable certificate.  It
uses the known simulator laws to supply the action variance and density
curvature.  It isolates the mathematical correction from the later problem of
calibrating those declarations.

## Frozen protocol

- Ten mechanisms from `sim.py`: linear, Friedman, sparse, high-frequency,
  heteroscedastic, t3, lognormal, discrete-cell, high-SNR and low-SNR.
- Seeds `41001` through `41010`; `n=4,000`, with the first half used for all
  fitted geometry and score-cap choices and the second half used once for the
  certificate.
- Losses: MAE and Q90 pinball in their existing package scaling.
- Geometry: learned one-dimensional outcome-blind key and raw standardized
  geometry with an independently learned location alignment.
- Block size: `k=8`.
- Score cap: 1.5 times the 0.99 empirical quantile of matching learning-half
  block scores.  No certification-half score enters the cap.
- Concentration: maximum of Bentkus and diversified-Kelly lower endpoints,
  each at alpha 0.025.  This is a two-route union, not post-selection at alpha
  0.05 for each route.
- First-order comparator: the already-proved barycentric coupled penalty.
- Second-order route: minimum of the loss-specific first-order action-MAD
  penalty and `kappa * action_variance / 2`, with the global simulator density
  envelope as `kappa`.

## Endpoints fixed before the run

Report, by loss and geometry:

1. violations of each lower endpoint against the exact fixed-design Bayes
   floor;
2. median bound/floor ratio;
3. paired strict wins of curvature over barycentric;
4. mechanism-level median wins.

No Mayo number is changed by this run.  A positive result advances the
conditional theorem route only; a production result additionally requires an
independent action-dispersion and curvature declaration.

