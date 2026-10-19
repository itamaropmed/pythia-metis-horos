# Registered refinement-test null calibration

Written after observing correlated rejection clusters in the 500-row power
battery and before opening the independent seeds below.

The preceding mechanisms reused each integer seed, so its 500 rows are not
500 independent null calibrations.  This follow-up uses:

- ten mechanisms;
- 100 disjoint seeds per mechanism (`660000 + 1000*m + repetition`);
- 512 entities per run;
- an independent nuisance refinement with 2, 4 or 8 states, rotated by
  repetition;
- 199 within-procedure permutations;
- rejection threshold 0.05.

Primary gate: the two-sided 95% Wilson interval for the 1,000-run rejection
rate contains 0.05 and has upper endpoint below 0.08.  Mechanism and nuisance
cardinality results are secondary diagnostics.  The mathematical guarantee
still rests on conditional exchangeability; simulation is a regression audit,
not its proof.
