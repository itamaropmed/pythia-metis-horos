# Clustered finite-cell confirmation registration

Date frozen: 2026-10-14, before running the confirmation seeds below.

## Question

Does the case-weighted finite-cell SAA lower endpoint retain one-sided
coverage when cases from the same patient are arbitrarily dependent, and does
externally declared cell-local support improve resolution over a matched
patient-cluster-valid global-range endpoint?

The concentration inequality is classical bounded differences.  The object
being evaluated is its composition with the downward-biased finite-cell SAA
functional, unequal patient sizes, local loss sensitivities, fail-closed
contracts, and the fixed-policy information-value architecture.  This study
does not register the underlying inequality as a new theorem.

## Frozen known-law battery

- Ten bounded discrete residual laws: Rademacher, center-heavy, tail-heavy,
  right-skew, left-skew, rare-tail, five-point uniform, bimodal-center,
  asymmetric mixture, and nine-point uniform.
- 100 fresh seeds per law: `760001` through `761000`, partitioned in blocks of
  100 by the order in the runner.
- 80 independent patients per draw and exactly 12 repeated cases per patient.
- One residual is sampled per patient and copied to every case, producing
  maximal within-patient outcome dependence.
- Eight frozen representation cells.  Cell locations differ by five units and
  residual scales increase deterministically with the cell index.
- The exact case-weighted conditional median risk is computed analytically
  from the discrete law and the observed cell weights.
- Confidence level: 95% (`alpha=0.05`).

Registered endpoints:

1. patient-cluster-valid Peras with analytic cell-local ranges;
2. patient-cluster-valid finite-cell SAA with one analytic global range;
3. a row-IID calculation, reported only as an invalid negative control.

Primary gates:

1. no local-cluster endpoint may exceed its known true risk;
2. the Wilson 95% lower limit for coverage must be at least 0.95;
3. report positive-endpoint rate and median bound/truth without hiding zeros;
4. report strict local-range wins over the global-range cluster endpoint,
   including all ten mechanism medians.

## Frozen dependence stress test

- 1,000 fresh seeds `880001` through `881000`.
- Center-heavy residual law on one cell, 60 independent patients and 100
  identical repeated cases per patient.
- The true MAE floor is exactly 0.4.
- Compare the patient-cluster endpoint with the deliberately invalid row-IID
  shortcut.  The row calculation is not a comparator eligible to win; it is a
  falsification control expected to become overconfident as rows are repeated.

The raw aggregate-safe output is written as JSON lines.  No result-dependent
route selection is permitted after this registration.
