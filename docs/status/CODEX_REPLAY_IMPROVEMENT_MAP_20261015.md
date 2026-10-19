# Codex replay: what failed, what changed in Pythia, and how each fix is tested

**Purpose.** Re-running Codex-style suites on similar cases should now succeed. Each row below links a failure recorded in `BOUND_BEHAVIOR_CATALOG_20261015.md` to the change that addresses it, and to the frozen test that checks it.

| Codex failure | Root cause | Pythia change | Test (frozen) | Status |
|---|---|---|---|---|
| Continuous MAE suite: pair/group bound at ≈0.49 of the floor | First-order drift penalty forces pairs; pair optimism (0.71 for MAE) | **Smooth domination:** second-order penalty, large blocks, domination-Bentkus plus block-dKelly portfolio | `continuous_20261015` regime I | Dev: wins 30/30 in 9 of 11 designs (MAE and Q90) |
| Paired Bentkus beat adaptive groups on 543/800 pinball cells | Group drift penalty was linear in diameter | Second-order penalty. The portfolio contains the pair route (k = 2) as a special case | Same suite, Q90 | Dev Q90: 0.53–0.75 of the floor vs ≤ 0.48 for the best accepted route |
| Sparse 10D/12D bound = 0 | Isotropic metric diluted by irrelevant coordinates | Declared per-coordinate moduli (l1 metric), plus a frozen linear alignment chosen by a declared-modulus rule | Regime A, `sparse10` | Dev: 0.67 MAE / 0.38 Q90, against best accepted 0.58 / 0.22 |
| Linear 5D: all methods ≈ 0 for Q90 | Large block diameter in 5D | Alignment rule removes the linear signal (declared residual modulus 0.24 vs 2.0) | Regime A, `linear5` | Dev: Q90 0.68 |
| Repeated patients / shared shocks: row-iid shortcuts invalid | Ignored dependence | Cluster route; dependency-graph route with audit and K-curve | `graph_20261014` | Dev: graph 200/200 valid where row-iid gave 88–90.5% |
| Finite cells: block-SAA optimism | Concentration over blocks | Fixed-action domination over clusters | `domination_20261014` | **Confirmed** (MAE 0.900 / Q90 0.847 / window 0.934, 0 violations) |
| Contaminated Gaussian: tie with block route | Cap grid max 8 | Extend the cap grid; the alpha cost is ≈ log of the grid size | Next registration | Open |
| Heteroscedastic/skew Q90: curvature penalty ≈ 0 | Global sup density | Row-local curvature κ_j = f_max/s_j (already in the smooth route), with local density bounds | Regime A, `hetero1`/`skew1` | Dev: Q90 0.68 / 0.55 |
| Fragmented note cells | In-cell optimism | Equality lattice (Codex) plus domination | Mayo lattice | Exploratory |
| Hidden target channels | Outside the declared representation | Not fixable by any bound; needs a representation-completeness audit | — | Open (design) |
| kNN beat the witness centre as a point estimator | Estimator, not bound | Out of scope for this round (bound first) | — | Open |

## What does not change

- The declarations themselves: the modulus L or L_i and the density bound. Every accepted near-replicate or kNN lower bound needs the same declarations. Pythia runs under identical declarations and reports tipping points.
- Mayo stays retrospective and exploratory until a prospective cohort exists.
