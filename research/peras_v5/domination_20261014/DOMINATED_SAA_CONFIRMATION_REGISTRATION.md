# Fixed-action dominated full-sample SAA: confirmation registration

Frozen 2026-10-14. Frozen after development seeds `2,000,001`–`2,000,470` (20 per law) and the
stress seeds `2,500,001`+; frozen before opening confirmation seeds `2,200,001`–`2,200,500`.

## Claim under test
For independent entity clusters and outcome-blind finite cells, the exact full-sample truncated
SAA statistic is pathwise dominated by the mean of independent, bounded, unobserved cluster
losses at the Bayes actions; inverting Bentkus with the observed statistic and the common
cluster range lower-bounds the case-weighted Bayes risk with no outcome support, no blocks and
no predictor (theory: `docs/theory/DOMINATED_SAA_20261014.md`).

## Frozen design (identical to SUPPORT_FREE_CLUSTER_BLOCK_REGISTRATION.md)
Ten unbounded noise laws; 50 fresh seeds each; 4,096 patients copied to two cases; eight frozen
cells with scale `.7+.12c` and location `5c`.

## Frozen routes
- Dominated SAA: inequality **Bentkus only** (development: Hoeffding never selected);
  caps `(.25,.5,1,2,4,8)` in original loss units; family alpha .05 Bonferroni over 6 caps;
  all cases retained.
- Comparator 1: Codex support-free block-SAA portfolio, exactly as registered (5 block sizes x
  6 caps, dKelly, alpha .05).  Comparator 2 (MAE): disjoint within-cell pairs, dKelly, 6 caps.

## Losses
Primary: MAE.  Secondary (separate alpha .05 families, reported in full): pinball q=.9; window
miss with half-window 1.0 (caps not applicable).

## Primary gates (MAE)
1. zero endpoints above analytic gold; 2. Wilson 95% lower validity >= .95;
3. strict wins over block-SAA in a majority of runs and of mechanism medians;
4. report every loss, tie, cap selection and median bound/gold.
Development expectation, not a gate: contaminated Gaussian may tie/lose (cap grid max 8).
