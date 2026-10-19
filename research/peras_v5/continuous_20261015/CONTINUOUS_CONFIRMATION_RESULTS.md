# PYTHIA-S vs accepted pair/kNN lower bounds — confirmation results (2026-10-15)

Untouched seeds; frozen code (`FROZEN_CODE_SHA256.txt`); identical declarations for all methods.
Raw SHA-256: regime I `0b344545d4987a7a91d715e977aad16f49e991b554158a943fa138680a86b822`;
regime A `7b869ee1332c9462efea1d3ef9cba9d2b73e75c799297810b0e91fc1f474a226`,
`321e5ce52cc9c033259e0461e2e84b9ff775a3755fda19b9e0430f447c55c5ba`.

## Gate 1 (validity): passed
Zero endpoints above the known floor for every method, regime and loss (1,100 PYTHIA-S endpoints).

## Gate 2 (beat the best accepted route in a majority of runs, >= 8/11 designs): passed
| Regime | MAE designs won | Q90 designs won |
|---|---:|---:|
| I (isotropic L, n=512) | **9/11** | **9/11** |
| A (anisotropic L_i + frozen alignment rule, n=1024) | **10/11** | **11/11** |

Median bound/true floor (PYTHIA-S vs best of the accepted routes):
| Regime, loss | PYTHIA-S range over designs | Accepted standard (first-order pair) | Best accepted incl. upgraded pair |
|---|---|---|---|
| I, MAE | 0.64–0.78 (excl. linear5/sparse10) | 0.34–0.59 | 0.54–0.69 |
| I, Q90 | 0.32–0.74 | 0.00–0.27 | 0.10–0.48 |
| A, MAE | 0.65–0.80 | 0.38–0.61 | 0.56–0.76 |
| A, Q90 | 0.35–0.76 | 0.00–0.34 | 0.16–0.75 |

## Losses and ties (reported in full)
- Regime I `linear5` MAE: PYTHIA-S 0.473 vs upgraded pair (second-order, a Pythia ingredient) 0.487 — lost 0/30
  (alpha cost of the portfolio); vs the literature first-order pair 0.024 it wins.  Q90 linear5 and all of
  sparse10: every method 0 under the isotropic declaration.
- Regime A `skew1` MAE: 10/20 wins (median 0.707 vs kNN-group 0.708) — tie-level; `t3_2` MAE 13/20.
- These are the next targets: smaller ledgers for d>2, and a local-linear witness.

## Reading
Under the same declarations, the second-order smooth-domination certificate is materially tighter than the
accepted nearest-neighbour pair and kNN-group lower bounds, especially for Q90 (where the accepted routes
are often 0).  Isotropic declarations remain uninformative in 5-10 dimensions; per-coordinate moduli plus a
frozen alignment repair this, at the cost of a stronger, explicitly declared smoothness contract.
