# Geometry rule (PYTHIA-S-G) — confirmation results (2026-10-15)

Untouched seeds 7,000,001+/7,100,001+, 20 seeds x 11 designs x {MAE, Q90}; frozen code `FROZEN_GEOMETRY_SHA256.txt`
(later additive change: optional `local_moduli` argument, default None, behaviour unchanged).
Raw SHA-256: `a1517c334b140a89ee4040b5cbd224ef09b91b82d9b8f25685cddede9957449f`,
`64adadf93bc89cddf10768663f7bd8374e6240e0ee75a2b3ffdfad088222551e`.

**All three gates passed for both losses.**
1. Validity: 0 endpoints above the true floor (440 new, 440 frozen, all accepted routes).
2. Never worse than the frozen PYTHIA-S by >2% (min ratio 0.993 MAE, 0.995 Q90); >5% better in 3 designs per loss.
3. Beat the best accepted route (pair first/second order, kNN groups, on either alignment) in a majority of runs in
   **10/11** designs for both losses.

| Design | MAE frozen -> new | Q90 frozen -> new | Driver |
|---|---|---|---|
| quad2 | 0.658 -> **0.717** | 0.337 -> **0.562** (x1.67) | alignment chosen by smallest declared modulus |
| interact2 | 0.687 -> **0.736** | 0.483 -> **0.616** | same |
| sparse10 | 0.657 -> **0.695** | 0.364 -> **0.551** (x1.52) | balanced OT blocks |
| smooth2 / sine1 | +1 to +4% | +1 to +4% | kernel-ridge alignment |
| others | unchanged (+-1%) | unchanged | rule leaves easy designs alone |

Remaining weak spot: `skew1` (lognormal noise) — new 0.713/0.547 vs best accepted 0.713/0.570 (8/20 and 10/20 wins).
The global density bound of the lognormal is large, so the curvature penalty is weak; next fix is a local
density bound near the quantile (direction B2).
