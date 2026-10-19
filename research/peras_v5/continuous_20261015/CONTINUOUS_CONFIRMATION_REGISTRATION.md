# Smooth-domination (PERAS-S) vs accepted pair/kNN lower bounds: confirmation registration

Frozen on 2026-10-15. The configuration was frozen after these development runs:
- `dev512.jsonl`: seeds 4,000,001+;
- `dev2048.jsonl`: seeds 4,500,001+;
- `devA*.jsonl`: seeds 4,700,001+ and 4,800,001+.

It was frozen before opening confirmation seeds 5,000,001+ and 5,100,001+.

## Motivation (Codex replay)

Codex's continuous suites found:
- pair/group envelopes at about 0.49 of the floor;
- paired Bentkus beating adaptive groups on most pinball cells;
- sparse and high-dimensional designs returning 0.

This study replays that class of designs: ten smooth mechanisms plus a sparse 10-dimensional one; noise that is Gaussian, Student t3, skewed lognormal or bimodal; homoscedastic and heteroscedastic. It asks whether the new route fixes those failures.

## Fairness contract

Every method receives the same:
- rows;
- declared modulus (isotropic L, or per-coordinate L_i in regime A);
- density/curvature bound;
- cap grid `(.25, .5, 1, 2, 4, 8)`;
- family alpha .05.

Each accepted route also uses the better of Bentkus and dKelly (alpha split), Bonferroni over its own caps.

## Methods

- **pair_first (the accepted standard):** global greedy nearest-neighbour pairs, the pair SAA witness, and a first-order Lipschitz penalty.
- **pair_second:** the same, upgraded with the second-order penalty. This baseline is strictly stronger than the literature version.
- **knn_group_first:** recursive nearest-neighbour groups (k = 4–64), independent block scores, and a first-order penalty, with the best k Bonferroni-charged.
- **PERAS-S:**
  - geometry: kd-tree if d ≤ 2, recursive matching otherwise (outcome-blind rule);
  - k ∈ {2, …, 64};
  - all caps;
  - concentration ∈ {domination-Bentkus over all rows, block-dKelly};
  - second-order penalty;
  - every candidate charged.

## Regimes

- **Regime I** (isotropic L, n = 512, all rows certify): `run_continuous_suite.py 30 512 5000001`.
- **Regime A** (n = 1024):
  - split the rows into halves;
  - fit a frozen ridge-linear alignment on half A;
  - use it only if the declared residual modulus is below half the raw modulus;
  - use per-coordinate moduli with an l1 metric;
  - certify on half B: `run_aligned.py 20 1024 5100001` for the first six designs and `5200001` for the remaining five. Frozen code hashes: `FROZEN_CODE_SHA256.txt`.

## Gates

1. Zero endpoints above the known floor for PERAS-S, in each regime and loss.
2. PERAS-S strictly above the max of all accepted routes in a majority of runs, in at least 8 of the 11 designs, for each loss. The baseline max is unadjusted, which favours the baselines.
3. Report all ties, zeros and losses.

The study does not claim to beat every published method, only the accepted kNN/pair lower bounds run under identical declarations.
