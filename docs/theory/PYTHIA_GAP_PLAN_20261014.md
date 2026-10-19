# Pythia gap-closing plan (survey + development), 2026-10-14

This plan follows the Codex checkpoint at commit `9e57508`, branch `codex/pythia-support-free-cluster`.

## Verified state at handover

- **Test suite.** 334 passed, 8 skipped, re-run on the device in two shards: 297 + 37 passed; 1 + 7 skipped.
- **Strongest certificates:**
  - support-free clustered block-SAA: MAE median bound/gold 0.787, 500/500 valid;
  - fixed-policy value bridge;
  - finite-cell lattice with refusal semantics.
- **Open gap recorded by Codex** (`CLUSTER_CONCENTRATION_GAP_20261014.md`): there is no variance- or tail-sensitive concentration for the *nonlinear* full-sample SAA witness. As a result, Pythia pays either:
  - block optimism (block-SAA), or
  - a hard outcome range with McDiarmid (finite-cell SAA).

## Survey position (what modern work already gives)

**Stochastic-programming optimality-gap estimation** (Mak, Morton and Wood 1999; Bayraksan and Morton 2006; Shapiro, Dentcheva and Ruszczyński):
- **Known:** E[min SAA] ≤ true minimum, and fixed-candidate domination. Lower bounds for stochastic programs come from batches of SAA optima.
- **Gap for Pythia:** they are used as optimality-gap estimates, with batch-means CLT intervals. They are not finite-sample, support-free, cluster-valid Bayes-floor certificates over representation cells.

**Range-only concentration** (Hoeffding 1963; Bentkus 2004):
- **Known:** tails of sums of independent bounded variables that need only the mean and the range.
- **Gap for Pythia:** these tools can be applied to *unobserved* dominating sums. That is the key that unlocks full-sample SAA.

**Betting / empirical Bernstein** (Waudby-Smith and Ramdas 2024; Maurer and Pontil 2009):
- **Known:** variance-adaptive intervals.
- **Gap for Pythia:** they require the observations themselves, which are not available for the dominating sum.

**Weak-interaction Bernstein** (Maurer 2017/2018):
- **Known:** variance-sensitive bounds for nonlinear functions.
- **Gap for Pythia:** this is still the route to variance sensitivity for the SAA functional (phase 5).

**Bayes-error / residual-variance estimation** (Devroye, Györfi, Lugosi and Walk 2018; Noshad and Hero; Ishida et al. 2023):
- **Known:** consistent point estimators.
- **Gap for Pythia:** they are comparators for the estimate tier.

## Phases

1. **Fixed-action domination: full-sample SAA without blocks, support or a model** (this session).
   - **Mechanism:** the in-sample optimum is pathwise dominated by the loss at the unknown Bayes actions. That loss is a sum of independent bounded cluster terms, so Bentkus and Hoeffding invert it. There is no block optimism and the concentration runs over all G clusters.
   - **Coverage:** MAE, every pinball level and window-miss, with exact computation of the truncated optimum.
   - **Protocol:** development, then registration, then an untouched confirmation on the frozen support-free design; plus a validity stress test (small G, many cells, unequal clusters, shared cluster effects).
2. **Fragmentation accounting.** Under domination, refining cells never weakens concentration. Only cell-level optimism shrinks the witness. Quantify optimism against cell size, and expose a declared second-order pooling penalty (the curvature/Jensen theorem) for sparse fine cells.
3. **Mayo re-run (aggregate only, exploratory).** Run the family lattice with the dominated route, then the fixed-policy value bridge (L_coarse − U_policy) to check whether the zero value lower bound can become positive.
4. **Estimator tier.** Add author-faithful comparators: nearest-neighbour residual variance, Hero's benchmark estimators, the Anderson–Bjarnadottir attainable MAE.
5. **Variance sensitivity for the nonlinear functional.** A weak-interaction Bernstein for the cellwise optimum, starting from the first-order sensitivities the domination exposes.
6. **Drift.** A realized-population floor with time-local cells, and nowcast-then-verify.
