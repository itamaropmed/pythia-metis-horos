# Capped overlapping Q90 development registration

Development-only registry, written before opening seeds `136001--136100`:

- ten bounded exact-law mechanisms with analytic/numerical known Q90 risk;
- 800 independent entities per run;
- 20 outcome-blind balanced random partitions at each
  `k=(8,16,32,40,80)`;
- score caps `(.005,.01,.02,.04,.08)`;
- one `.05` family ledger across all 25 multi-point candidates;
- zero barycentric discrepancy because every entity has the same law;
- comparators: paired Bentkus and disjoint-block diversified Kelly, each with
  its own valid `.05` route.

This is a development suite.  Any route decision must be frozen and tested on
new seeds before it becomes evidence.  The registry tests exact-law
concentration/optimism only; a later near-replicate audit must test nonzero
barycentric penalties.
