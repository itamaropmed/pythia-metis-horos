# Repeated-environment action-dispersion calibration

## Purpose

The curvature theorem needs an upper bound on within-block variation of the
latent median or quantile actions.  One outcome per unit cannot identify that
quantity.  `pythia.core.action_dispersion_calibration` adds a finite-sample
route for experiments with repeated measurements of the same fixed units.

## Construction

For `E` independent environment rows and `m` used units, Pythia assigns
`alpha/(2m)` to each tail of each unit's scalar quantile interval.  Its ranks
come from the exact binomial distribution.  This remains conservative for
atoms because the probability below a lower quantile is at most `q` while the
probability at or below it is at least `q`.

On the joint event, every latent action belongs to its interval.  Within each
globally disjoint block, Pythia enumerates every vertex of the interval box and
maximizes:

- variance around the arithmetic centre;
- absolute dispersion around the median;
- maximum displacement from the arithmetic centre; and
- when registered unit curvature bounds are supplied, the optimized
  curvature-weighted quadratic penalty.

These objectives are convex in the action vector, so a maximum over a box is
attained at a vertex.  The implementation caps exact enumeration at block size
16 and refuses larger blocks rather than silently approximating the maximum.

## Contract and limitation

Rows must be mutually independent repeated environments measuring the same
fixed units.  Units may share arbitrary shocks within a row.  A hard outcome
support is required and automatically bounds the quantile action.  Curvature
provenance is separate from environment and support provenance.

This is a non-oracle calibration of action dispersion, but it can be very
conservative when there are many units because simultaneous unitwise coverage
spends a large multiplicity budget.  Sharper block-level confidence regions
are a research target; they must dominate this valid baseline on untouched
designs before replacement.

Mayo's current retrospective table contains one realized duration per case,
not repeated target environments for the same case.  The route therefore
refuses Mayo rather than borrowing information from a fitted outcome model.

## Tests

The test suite checks exact binomial tail spending, known-normal median action
dispersion, shared within-row shocks, capacity refusal, and missing independence
attestation.  A subsequently frozen ten-mechanism battery covered all four
exact targets in 1,000/1,000 rows (95% Wilson lower 0.9962). Median curvature
cap/truth was 1.221 for median actions and 1.284 for Q90 actions. See
`ACTION_DISPERSION_CONFIRMATION_RESULTS_20261014.md`; the weak-signal worst
ratios remain a tightening target.
