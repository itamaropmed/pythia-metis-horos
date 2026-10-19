# Registered equality-class estimator confirmation

Written after the bound-lattice confirmation and before opening seeds
`64001`--`64050`.

## Estimator rule

When two frozen representations have an independently attested equal Bayes
risk, route the empirical SAA floor estimate to the member with the fewest
observed cells; break ties by frozen registry order.  Labels and registry order
are visible without certification outcomes.

The rationale is one-sided.  Refining an empirical decision table can only
decrease its in-sample optimum.  If the refinement is known to leave the
population optimum unchanged, the finer SAA plug-in has weakly greater
downward bias in expectation.  This does not prove uniformly smaller absolute
or squared error, so the latter is a confirmation endpoint rather than a
theorem claim.

## Frozen confirmation

- Same ten mechanisms and five-node hierarchy as the family-lattice battery.
- 50 fresh seeds per mechanism, 2,048 entities per seed.
- Primary comparison: absolute error of the raw nuisance-fragmented SAA score
  versus the routed equality-class estimate.
- Primary endpoints:
  1. routed estimator has smaller absolute error in a majority of 500 runs;
  2. median error improves in a majority of the ten mechanisms;
  3. the routing source is the canonical neighborhood in all 500 runs;
  4. all previous bound-validity and order endpoints remain satisfied.

Mechanisms are the scientific unit.  No point-estimator SOTA claim follows
from this synthetic equality setting alone.
