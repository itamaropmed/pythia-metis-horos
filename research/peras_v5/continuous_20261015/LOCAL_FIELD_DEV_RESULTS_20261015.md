# Development: local modulus field (B1) and local curvature (B2) — not confirmations

B1 local modulus field (per-block declared moduli = sup of the residual action gradient over the block box,
simulator truth x1.05), 8 seeds x 6 designs, n=1024: 0 violations; strict gain 8/8 wherever the gradient varies:
| Design | MAE global -> local | Q90 global -> local |
|---|---|---|
| quad2 | 0.718 -> 0.741 | 0.575 -> 0.626 |
| smooth2 | 0.736 -> 0.760 | 0.636 -> 0.666 |
| interact2 | 0.727 -> 0.742 | 0.611 -> 0.642 |
| sine1 | 0.769 -> 0.776 | 0.665 -> 0.672 |
| linear1, hetero1 | unchanged (constant gradient) | unchanged |
Tier: conditional on a declared local field; in practice it must be declared/calibrated (tipping point reported).

B2 local curvature near the quantile: valid, wins 8/8 in most designs but gains are < 0.001 of the floor —
NEGATIVE result.  The skew1 weakness is not the curvature penalty; it comes from truncation/concentration
under the heavy right tail (to be diagnosed next: cap grid, variance-adaptive concentration).
