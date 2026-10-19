# Response to the ICML-reader critique (2026-09-14)

The pasted judge's central criticism is right: the selected Bayes action's
Lipschitz cap is load-bearing. The paper now gives a concrete *sufficient*
quantile route (conditional-cdf difference `K d`, density lower bound `m`,
therefore `L=K/m`) and states that Mayo has no independent audit of either
constant. The window-maximizing action may jump at a mode switch, so the
quantile route cannot certify W30. The clinical calculation is explicitly an
illustrative sensitivity result, not a statement that one fifth of future
cases are irreducibly mis-scheduled.

The review also missed two mathematical/paper-code discrepancies:

1. The old grouped pinball formula in the paper was a symmetric
   `min(tau,1-tau)` relaxation; the implementation uses the sharper
   *asymmetric* pinball relaxation. The revised proof now matches code.
2. `tau*(1-tau)*R` is a valid universal group-score cap, but a fixed group
   of size `k` can have a smaller sharp cap. The revision states and proves
   `R max_m min(tau*(k-m)/k, (1-tau)*m/k)`.

The old shared-bias proposition inferred a Bayes-risk lower bound from MAD
subadditivity without an independently valid bridge from predictor error to
Bayes risk. It has been replaced by a precise non-identifiability
counterexample. The paper also corrects “any decision rule” to “a selected
Bayes action,” “deployed predictor” to “frozen OOF comparator,” and the
mislabeling of the legacy Mayo calculation as the later frozen geometry route.

The positive result is now inspectable in the supplementary ten-row table:
on seed 20260928, 20 draws for each of ten analytic-noise MAE designs, the
group bound is larger than the paired bound in 180/200 draws, ties in all 20
sparse-20D draws, and is below known risk in all 200. The pooled median
usable tightness is 0.6405 versus 0.5634. This is a win over the *named
paired baseline*; it is not a claim of global best estimator or bound.
The separate five-loss results and later geometry route are included with
their exact comparators, zero-floor ties, and retrospective selection caveat.

The paper's distinct target is a finite-design lower confidence bound on
attainable decision risk. Its relation to
[Anderson--Bjarnadottir (2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296904),
[Bound to Disagree (2026)](https://proceedings.mlr.press/v337/bazinet26a.html),
and [Model Agreement via Anchoring (2026)](https://proceedings.mlr.press/v336/eaton26a.html)
is now explicit. This narrows, but does not settle, a publishable-priority
claim; an independent systematic related-work and proof audit remains due.

Finally, the file called “ICML-formatted” is not actually in the official
ICML style: it uses a one-column article class, displays the author, and has
not been checked against the main-body page limit. The revised PDF is a
reader draft. The [official ICML 2026 instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
require the conference style, anonymization for submission, and an eight-page
main-body limit. A real submission must migrate to the appropriate year's
official template and pass its format check.
