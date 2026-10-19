# Geometry rule confirmation registration (bound tightness)

Frozen 2026-10-15 after development seeds 6,000,001+/6,100,001+ (`GEOMETRY_DEV_RESULTS_20261015.md`), before
opening seeds 7,000,001+ (designs 1-6) and 7,100,001+ (designs 7-11).  Regime A design, n=1024, 20 seeds.

Frozen rule (PERAS-S-G): alignment = argmin of the declared per-coordinate residual modulus sum over
{none, ridge-linear, kernel-ridge RBF(gamma=2/d, alpha=1)} fitted on half A; geometry = kd if the TwoNN dimension of
the modulus-scaled certification features is < 3, else {matching, balanced OT}; block sizes, caps, alpha,
concentration portfolio and penalty unchanged from PERAS-S.

Comparators: frozen PERAS-S regime A (previous confirmation); accepted pair (first order), upgraded pair (second
order) and kNN group, each run both on the old and on the new alignment (best taken; favours baselines).

Gates (each loss): (1) zero endpoints above the true floor; (2) PERAS-S-G median >= 0.98 x frozen PERAS-S median in
every design and > 1.05 x in at least 3 designs; (3) PERAS-S-G strictly above the best accepted route (on either
alignment) in a majority of runs in >= 9/11 designs.  All losses/ties reported.
