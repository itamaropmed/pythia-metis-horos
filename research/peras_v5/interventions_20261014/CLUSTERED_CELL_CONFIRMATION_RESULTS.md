# Clustered finite-cell confirmation results

Date: 2026-10-14

This is the untouched confirmation registered in
`CLUSTERED_CELL_CONFIRMATION_REGISTRATION.md`.

## Primary known-law battery

- Patient-cluster-valid local endpoint: **1,000 / 1,000 valid**; Wilson 95%
  lower coverage limit **0.9962**.
- Patient-cluster-valid global endpoint: **1,000 / 1,000 valid**.
- Positive local endpoints: **867 / 1,000**.
- Strict local-range wins over the cluster-valid global-range endpoint:
  **867 / 1,000**.
- Median local bound / true floor: **0.36594**.
- Median global-range bound / true floor: **0.00000**.
- The row-IID negative control exceeded the known floor in 9 / 1,000 runs.

| Noise law | Median true floor | Local cluster bound | Global cluster bound | Invalid row-IID bound | Local validity | Strict local wins |
|---|---:|---:|---:|---:|---:|---:|
| Rademacher | 1.1230 | 0.5250 | 0 | 0.7513 | 100/100 | 100/100 |
| Center-heavy | 0.4468 | 0.1416 | 0 | 0.3667 | 100/100 | 97/100 |
| Tail-heavy | 1.0067 | 0.5367 | 0 | 0.7578 | 100/100 | 100/100 |
| Right-skew | 0.3936 | 0.0598 | 0 | 0.2843 | 100/100 | 89/100 |
| Left-skew | 0.3915 | 0.0500 | 0 | 0.2744 | 100/100 | 81/100 |
| Rare-tail | 0.1119 | 0 | 0 | 0.0222 | 100/100 | 0/100 |
| Five-point uniform | 0.6716 | 0.3142 | 0 | 0.5378 | 100/100 | 100/100 |
| Bimodal-center | 0.9774 | 0.4570 | 0 | 0.6795 | 100/100 | 100/100 |
| Asymmetric mixture | 0.4826 | 0.1340 | 0 | 0.3470 | 100/100 | 100/100 |
| Nine-point uniform | 0.6222 | 0.2476 | 0 | 0.4733 | 100/100 | 100/100 |

The global comparator is zero because its honest support must span the
between-cell location shifts.  The local method removes those irrelevant
shifts only because the conditional support map was supplied analytically.
The rare-tail mechanism is the important unresolved failure: a valid local
range is not enough to resolve a very small floor from 80 independent
patients.

## Dependence falsification

The separate stress test copied one center-heavy residual to 100 rows per
patient.

| Route | Valid draws | Wilson 95% interval | Median lower endpoint |
|---|---:|---:|---:|
| Patient-cluster Pythia | **1,000 / 1,000** | 0.9962--1.0000 | 0.0840 |
| Row-IID shortcut (invalid) | **638 / 1,000** | 0.6077--0.6672 | 0.3684 |

Row duplication therefore created a much tighter-looking endpoint while
destroying nominal coverage.  This is the intended reason the public API
refuses to treat repeated cases as independent.

## Interpretation

The result validates the current composition on these ten bounded laws.  It
does not make bounded differences novel, establish universal tightness, or
solve temporal dependence.  The next technical gap is a variance-sensitive
concentration route for the nonlinear SAA functional under unequal independent
cluster contributions.  Additive Bentkus, ordinary empirical Bernstein, and
bounded-mean betting cannot simply be substituted without proving that
transfer.

Raw JSONL SHA-256:
`6ec0025ece98962d68a5e5de8add74a1e414502b54accd8559d69d1326ebbb16`
