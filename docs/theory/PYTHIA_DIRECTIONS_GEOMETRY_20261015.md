# Directions to add to Pythia: geometry, topology, information, and system (2026-10-15)

Rule that frames every direction: an outcome-blind partition, metric, ordering or alignment fitted on independent
data can change only **tightness**, never **validity**.  New *assumptions* (declared moduli, dependence graphs,
curvature) change the claim tier and must carry provenance and tipping points.

## A. Geometry of the bound (tightness, validity-safe)
| # | Direction | Mechanism | Status |
|---|---|---|---|
| A1 | Balanced OT / balanced k-means blocks | minimise sum d^2 at exact block size = the second-order penalty | implemented (`balanced_blocks`), dev |
| A2 | Intrinsic-dimension planning and adaptive block sizes | E d^2 ~ (k/n)^(2/d_int); TwoNN/MLE on dev split; variable-size cells | planned |
| A3 | Smooth frozen alignment (kernel ridge / polynomial) chosen by smallest declared residual modulus | removes smooth signal before blocking | implemented in dev runner |
| A4 | Local-linear (Hessian) witness | penalty H^2 d^4 instead of L^2 d^2; exact truncated min along 1-D keys | planned (computational open problem) |
| A5 | Graph / manifold geometry: geodesic kNN distances, Ollivier-Ricci bottlenecks | blocks must not cross bottlenecks | planned |

## B. Smoothness and curvature declarations (tier: conditional/calibrated)
| B1 | Local modulus field L(x) | penalty sum L(x_j)^2 d_j^2; calibrated on dev, falsified one-sidedly | planned |
| B2 | Local density/curvature near the quantile | fixes hetero/skew Q90 | partly (row-local scale) |
| B3 | Wasserstein-regression calibration of conditional-law drift | principled source for L(x) | research |

## C. Topology (proposers, never certificates)
| C1 | Mapper multiscale cover as the representation lattice | outcome-blind strata at several resolutions; persistence picks scales | planned |
| C2 | Persistent-homology stability of note neighbourhoods | downgrade unstable communities | planned |

## D. Statistics of the bound
| D1 | Variance-adaptive concentration for the nonlinear SAA (weak-interaction Bernstein) | biggest remaining width gain | open theorem |
| D2 | Fractional colourings / LP for dependency graphs | n/chi* instead of n/K | planned |
| D3 | Anytime-valid (e-process) version for sequential Mayo monitoring and Metis search | | planned |

## E. Information and representation
| E1 | Leakage / completeness scanner (counts, digits, lengths, rounding, number words) | refuse + diagnostic floor ~0 | planned |
| E2 | Information value with support-free policy upper bounds | positive value at procedure level | planned |
| E3 | Drift: realized-period floor tracking, nowcast-then-verify | | partly (graph theorem) |

## F. System
| F1 | Point-estimate tier wrapping strong external estimators, bound as sanity floor | | planned |
| F2 | Metis: automated, ledgered search over the above choices | | architecture |
