# MAE witness-family theory and second OOD audit

## What changed

`mae_pair_group_envelope` is a new, train-free fixed-design MAE lower-bound
family. It takes a predeclared outcome-blind pair partition, a registry of
outcome-blind group partitions, clipping thresholds, a metric, and a conditional
median Lipschitz cap. It returns the maximum paired/grouped witness **after**
allocating total alpha across both families and all candidate methods and clips.
Prior tables reported each family at its own 95% level; picking the better of
those two after seeing outcomes did not retain 95% simultaneous coverage.

### Conditional theorem

Condition on the evaluated covariates and on pair/group constructions fixed
without their outcomes. Assume distinct sampling entities have conditionally
independent outcomes and there exists a selected conditional median `m(x)`
with `|m(x)-m(x')| <= L d(x,x')` for the declared mean-L1 metric. Let `R` be
the average conditional Bayes MAE on **all** audited entities.

For each pair `(i,j)`, the triangle inequality gives
`(|Yi-Yj|-L*d(Xi,Xj))/2 <= (|Yi-m(Xi)|+|Yj-m(Xj)|)/2` in expectation.
Clipping the nonnegative difference term before the deterministic penalty
preserves the lower inequality. For each group, the intervals
`[m(Xi)-L*d(Xi,c), m(Xi)+L*d(Xi,c)]` have a common point `a` because the
median cap and metric triangle inequality make them pairwise intersect.
Minimizing the group's lower witness over all scalar `a` can only decrease its
expectation relative to this feasible `a`; its expectation is therefore at
most the group's average Bayes MAE. Pair and group scores are each bounded
after clipping, and disjoint groups are conditionally independent. One-sided
Hoeffding or independent-variable empirical-Bernstein radii give a lower
confidence event for each candidate. A union bound over **both families**,
their concentration methods, clipping thresholds, and group sizes gives
`P(max_candidate_lower_bound <= R | X) >= 1-alpha`.

This is a conditional theorem, not a data-free certificate. It does **not**
validate a median cap, independence, provenance, or a projection selected with
evaluation outcomes. The point estimates in the comparison have no analogous
finite-sample correctness theorem. Empirical-Bernstein uses [Maurer and
Pontil's independent bounded-variable result](https://arxiv.org/abs/0907.3740).

## New untouched synthetic challenge

The ten new generator names and analytic noise risks are frozen in
`horos/benchmark/ood_mae_designs_second.py`; they do not overlap the first ten.
The aggregate-only result is in `docs/results/ood_mae_second_20260913.json`.
At 512 entities and 20 independent draws/design, all 200 selection-safe bounds
were below the known conditional Bayes risk (95% Wilson lower coverage bound
98.1%). The combined median usable tightness was 48.6%, versus 41.8% paired
and 49.5% grouped at their *separate* 95% levels. The combined price is real:
it need not beat the best single family numerically.
Against paired alone at the same total 95% confidence, combined won 175
draws, lost 5, and tied 20 (within-suite one-sided sign `p≈1.0e-45`). This
is a substantial gain over the particular paired baseline on this frozen
challenge, not evidence of dominance over all published methods or future
distributions.

The estimator did **not** win at 512: train-free group jackknife median absolute
error was 0.0234, external kNN OOF 0.0110. At 192 entities, train-free LP
jackknife had median error 0.0157 versus kNN 0.0222, but draw-level wins were
99 versus 101. That is not significant superiority or a universal estimator
claim. The LP result remains exploratory, and `omniscient_lp_raw` is a
comparator rather than a certificate. The [Anderson--Bjarnadottir MAE-floor
heuristic](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296904)
also should not be equated with a covered lower bound.

The sparse 12-D design exposed a gap that cannot be repaired by post-hoc
tuning: all 20 full-metric bounds were zero. On those *same* generator draws,
an oracle, predeclared one-coordinate projection was positive in all 20 and
had 56.4% median tightness. This diagnoses metric dilution. Such a projection
is valid only if an independent source justifies that the conditional median
depends on the retained information and supplies its Lipschitz cap. Outcome
selection on the evaluation set would invalidate it. The familiar
[nonparametric dimension penalty](https://doi.org/10.1214/aos/1176345969)
is a reason to expect this difficulty, not a proof that any specific projected
metric is valid.

## Mayo isolation and augmentation

The historical Arm-B guarded values are locked in
`docs/results/mayo_frozen_ci_anchor.json` and recomputed by the private-data
test when the hashed Mayo artifacts are available: best OOF MAE 32.8700117,
floor estimate 31.1186421, calibrated stratified bound 28.8495996, and
conditional Q-cap rung 24.1867259. Public CI checks source/manifest hashes
and these aggregate values; it cannot recompute the clinical analysis without
the private files. The authored workflow installs the pinned `requirements.lock`
on Python 3.13 and reruns the second 200-draw known-risk regression. The 28.85
result remains **calibrated**, not proved.

The 2,048-patient in-memory target-injection falsification found baseline OOF
MAE 32.1905 on that subsample; 10%, 50%, and 100% note injection yielded
29.0118, 16.8501, and 0.0 mixed MAE. Full injection yielded a correctly
guarded pair bound 0 and estimate approximately 0, and the API refused the
post-outcome feature. Copying an existing schedule feature 20 times changed
the unaudited-L=300 pseudo-bound from 22.2148 to 20.2881 despite no increase
in true information. This is expected metric sensitivity, not evidence that
feature duplication harmed the Bayes information. No modified note or patient
row was persisted. These Mayo checks are retrospective diagnostics only.

## Remaining gates

1. An independent representation/metric audit is needed for a nonzero sparse
   certificate; Pythia must refuse or mark uninformative when that cannot be
   justified.
2. A train-free point estimator still has no uniform dominance, and a bound's
   coverage does not confer estimator accuracy. Test estimator diagnostics or
   abstention before advertising a production point estimate.
3. Mayo's private-data recomputation must run in a controlled private CI job;
   public GitHub CI can only protect the aggregate lock. The repository has no
   configured Git remote or committed revision yet, so the authored workflow
   is not running on GitHub.
