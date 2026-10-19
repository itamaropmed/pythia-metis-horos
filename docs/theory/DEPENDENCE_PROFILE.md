# Pythia Dependence Profile — theorem-candidate specification

## Status

The implementation in `pythia/core/dependence.py` is an **exploratory sensitivity
diagnostic**. It is never used automatically to upgrade a certificate. A proof audit and
finite-sample treatment are required before it can become a conditional certificate.

## Object

For squared loss, observed predictor risks and pairwise squared disagreements determine an
affine candidate covariance matrix for model errors at each proposed Bayes risk. The profile
searches for the smallest risk compatible with a positive-semidefinite covariance matrix and
a cap `lambda_max` on positive pairwise error correlation.

The report contains the full `B(lambda_max)` curve, the feasible dependence and risk-floor
intervals, active PSD/correlation constraints, and the analytically implied point at which the
implemented bound becomes vacuous. At `lambda_max=0`, the public curve is anchored to the
existing max-pair orthogonality rung. Allowing more dependence can never tighten the curve.
If no requested dependence cap admits a PSD moment matrix, the profile returns `refused`.

## Claim boundary

- This is not an estimator of the unknown dependence.
- The failed single-lambda fit in `folddep.py` remains a negative control.
- The empirical OOF moment matrix is not the population moment matrix; sampling uncertainty
  still needs simultaneous confidence bounds.
- No MSE proof is transferred to MAE. Any MAE dependence profile remains calibrated only.

## Literature audit required before novelty claim

- Bazinet, Zantedeschi & Germain (2026), *Bound to Disagree*:
  https://proceedings.mlr.press/v337/bazinet26a.html
- Eaton et al. (2026), *Model Agreement via Anchoring*:
  https://proceedings.mlr.press/v336/eaton26a.html
- Loureiro et al. (2022), correlated learner fluctuations:
  https://proceedings.mlr.press/v162/loureiro22a.html

The proposed distinction is a sensitivity analysis for the attainable/Bayes floor under
bounded shared error, rather than a bound on an already-trained target model's ordinary
generalisation gap. That distinction must survive a systematic literature review.
