# Frozen Q90 partition-mixture dKelly confirmation

Written before opening seeds `137001--137100`.

Development seeds showed that entity-McDiarmid overlapping witnesses beat
paired Bentkus but lost to disjoint dKelly.  A valid alternative averages—not
multiplies—the dKelly e-values from overlapping balanced partitions.  On a
development subset, `k=16` and score cap `.08` improved the median over a
matched single partition in all ten mechanisms and won 604/1,000 runs.

Frozen confirmation:

- the same ten bounded exact-law mechanisms and 800 entities;
- Q90, clip `[0,1]`, zero discrepancy;
- 20 registered balanced random partitions;
- exactly one candidate: `k=16`, score cap `.08`, alpha `.05`;
- matched comparators: the first partition's dKelly endpoint and paired
  Bentkus, each at alpha `.05`;
- primary endpoint: validity against known Q90 risk;
- superiority summaries: run wins, mechanism-median wins, and median
  bound/gold.  Mechanisms, not 1,000 conditionally related rows, are the
  conservative descriptive inference unit.

This confirmation tests exact-law concentration.  Nonzero partition-specific
barycentric penalties remain a separate future audit.
