# Decision-loss information and two-point risk bounds (2026-09-12)

## Scope and novelty boundary

The relevant information quantity is **decision-specific Bayes-risk reduction**:

\[
I_\ell(Z;Y\mid X)=R^*_{\ell}(X)-R^*_{\ell}(X,Z),\qquad
R^*_{\ell}(X)=\inf_a {\mathbb E}\,\ell(Y,a(X)).
\]

If `(X,Z)` contains `X`, then `I_ell >= 0` for every loss and action set.
This is the established Blackwell/decision-theoretic order, **not** a newly
invented Shannon information measure. The implementation of risk-interval
closure is already in `pythia/information_order.py`. It requires the same
population, loss, endpoint, timing, and simultaneous intervals; it must not
interpret a worse fitted estimator for rich features as negative information.

The development here is a reusable **task-conditioned two-point uncertainty
witness** and its implementation across MAE, MSE, pinball loss, and within-window
miss probability. It is a proof-audit candidate, not a literature-vetted novel
theorem claim.

## Conditional finite-audit theorem

Fix observed covariates `x_1,...,x_2n` and an outcome-blind disjoint pairing.
Let `a*_ell(x)` be a selected conditional Bayes action for loss `ell`, and
assume `|a*_ell(x)-a*_ell(x')| <= L d(x,x')` for every audited pair. Define

\[
h_\ell(y,y';\delta)=\frac12\inf_{|a-a'|\le\delta}
    \{\ell(y,a)+\ell(y',a')\},\qquad \delta=Ld(x,x').
\]

For each realized outcome pair, the optimal conditional actions are feasible
in the infimum. Hence the average conditional Bayes risk over all `2n`
endpoints is at least the average expectation of `h_ell` across the `n`
pairs. This deterministic inequality needs no model training and no
assumption that model errors are orthogonal. Conditional independence of
patient outcomes, a fixed outcome-blind matching, and bounded pair scores are
*additional sampling assumptions* used for Hoeffding or empirical-Bernstein
concentration. An `alpha` lower-confidence bound on that witness is thus a
conditional lower bound on the finite audited conditional Bayes risk.
It is not automatically a population, prospective, or representation-universal
bound. If patients share outcome shocks, use an externally justified block
concentration design instead of pair-level concentration.

Pointwise witnesses for scalar actions (`D=|y-y'|`, `delta=Ld`):

| Loss/risk | Two-point witness `h_ell` | Selected Bayes action |
|---|---:|---|
| MAE | `(D-delta)_+/2` | conditional median |
| MSE | `(D-delta)_+^2/4` | conditional mean |
| pinball at `tau` | `min(tau,1-tau)(D-delta)_+/2` | conditional `tau` quantile |
| miss outside `w` | `1{D>2w+delta}/2` | center maximizing within-`w` coverage |

The last witness is a safe pointwise lower bound. It is not necessarily the
exact value of the infimum for every configuration. W30 uses `w=30` minutes;
its complementary quantity `100*(1-bound)` is an **upper** bound on the
attainable percentage within 30 minutes. Simultaneous window-grid bounds can
also lower-bound absolute-error quantiles: if the lower miss-risk bound at `w`
exceeds `1-p`, every eligible predictor has its `p`-quantile absolute error
strictly greater than `w`.

The pair witness's corrected center is a **lower-bound plug-in**, not a
generally unbiased or optimal point estimator of Bayes risk. The concentration
bound has coverage under its conditions; the point estimate does not inherit
the same guarantee. Recovering a sharp point estimator requires extra
structure or separate validation against known-risk baselines.

## Multi-outcome group extension

The group method in `pythia/core/group_witness.py` uses a disjoint,
outcome-blind hierarchy of `2, 4, 8, 16` similar cases. Assume selected
Bayes actions at *observed* feature rows obey the pairwise Lipschitz cap.
For a group centroid `c`, set `delta_i=L d(x_i,c)`. The intervals
`[a*(x_i)-delta_i,a*(x_i)+delta_i]` intersect: for any two rows,
`|a*(x_i)-a*(x_j)| <= Ld(x_i,x_j) <= delta_i+delta_j`, and intervals on the
real line have the pairwise-intersection property. Choose a common **virtual**
action `a` in their intersection. No conditional Bayes action at the centroid
is needed; this matters when a fractional one-hot centroid is not a feasible
clinical case. That latent action gives the relaxed lower witness

\[
H_G=\frac1{|G|}\inf_a\sum_{i\in G}
\inf_{|a_i-a|\le\delta_i}\ell(y_i,a_i).
\]

The same pointwise argument establishes that the average conditional Bayes
risk over the fixed group endpoints exceeds `E[H_G]`. For pinball loss,
the inner loss is the pinball cost outside the interval around `a` of
radius `delta_i`; minimizing over breakpoints is exact. For window miss,
`H_G` is one minus the maximum overlap fraction of expanded outcome
intervals `[y_i-w-delta_i, y_i+w+delta_i]`. Independent group scores
concentrate by Hoeffding or empirical Bernstein; the predeclared group-size
and concentration registry is Bonferroni-protected. No predictor is fitted
inside Pythia. The feature metric and grouping still need outcome-blind
provenance, and the action smoothness requirement is a substantive condition.

## Ten known-noise designs, frozen seed

Run `python3 scripts/run_decision_loss_comparison.py` for ten DGP families,
20 draws each, 2,000 entities per draw, both `tau=.9` pinball and window
miss at `w=.1` in normalized outcome units. Truth is computed analytically
at the actual audited covariates, including heteroskedastic conditional
truth. For its pinball quantile, the heteroskedastic center cap is `L=1.48`,
not the median's `L=1.0`.

The 400/400 Hoeffding and 400/400 empirical-Bernstein lower bounds were below
known risk. There were only **200 independent draws**, because each draw
produced two correlated loss results; the draw-clustered 95% Wilson lower
coverage is **0.9812**, not the invalid result-level 0.9905. Empirical Bernstein
was strictly tighter in 200/400 runs, Hoeffding in 160/400, and 40 tied.
Pinball favored empirical Bernstein in 180/200; window miss favored Hoeffding
in 160/200. Median usable tightness was 0.319 for empirical Bernstein versus
0.253 for Hoeffding across losses. **Sparse ten-dimensional data produced
zero bounds in both methods.** These are exploratory stress tests, not evidence
that Pythia is universally tightest or has the best estimator. The pair center
was notably biased low; its median absolute error was 0.045 in the pooled
dimensionless benchmark. The previous MAE/MSE clipped comparison similarly
has counterexamples. Methods must be compared at the same confidence,
assumptions, estimand, and held-out design registry before any ranking claim.
An external leave-one-out 51-neighbor plug-in, fitted *outside Pythia*, was
closer to known Bayes risk in 366/400 trials, versus 31/400 for the pair
center (three ties). Its median absolute error was 0.00358, versus 0.045 for
the pair center. This is a decisive negative result for calling the pair
center a competitive point estimator on these DGPs; do not silently promote
it to a floor estimate. The kNN plug-in is an estimator, not a lower bound,
and it need not work in sparse/high-dimensional or shifted clinical settings.
An explicitly Bonferroni-protected two-method selector is implemented. It
also had 400/400 valid bounds, but its median usable tightness was 0.313,
below empirical Bernstein alone (0.319): method selection costs confidence
budget and cannot be advertised as a free improvement. It tied or beat the
better unsplit nominal-`alpha` method in only 40/400 trials.

On an independent second seed (`20260921`) of the same ten predeclared
families, the **selection-safe grouped bound** was below known risk in
400/400 trials (200/200 independent draws jointly valid; clustered 95%
Wilson lower coverage **0.9812**), with median usable
tightness **0.443** versus **0.315** for the selection-safe paired bound.
Grouped was strictly tighter in **288/400** and paired in **72/400**;
40 tied. At the design level, grouped won every trial in linear uniform,
nonlinear sine, heteroskedastic, quadratic, interaction, and bimodal noise;
it won 39/40 asymmetric-mixture trials. Paired was better on bounded heavy
tails and most high-dimensional trials, while both were zero on sparse 10D.
This is a substantial but **not universal** bound improvement. It is only a
comparison with the implemented matched-pair Hoeffding/Bernstein methods,
not every published lower-bound method. Repeated simulated draws and a
second seed do not turn this into a proof of method dominance.

Selecting the better *family* after observing outcomes requires another
split of the confidence budget. With `alpha=.025` for the paired family and
`alpha=.025` for the grouped family, the combined 95% selector was valid in
400/400 second-seed trials and had median usable tightness **0.418** versus
**0.315** for the paired-only 95% selector. Its design-level median was
higher in seven families, lower in high-dimensional and heavy-tail designs,
and tied at zero in sparse 10D. Do not take the unadjusted maximum of the two
separate 95% family bounds and call it 95%.

The grouped corrected center improved on the pair center in 312/400 second-
seed trials but was still not a competitive point estimator: median absolute
error **0.0369**, versus **0.00356** for the external kNN plug-in. The no-model-
training constraint inside Pythia is preserved, but a claim that Pythia has the
best estimator remains unsupported.
A train-free exploratory jackknife of unguarded local empirical Bayes scores
also lost: its second-seed median absolute error was **0.01044**, versus
**0.00356** for kNN, and it was closer than kNN in only 84/400 trials.
This jackknife is kept in the benchmark as a negative control, not released as
a Pythia estimator or certified bound.

A small-n, ten-design MAE comparison is now runnable with
`python3 scripts/run_omniscient_lp_comparison.py`. It solves a global
smoothness-constrained LP using all pairs and the same declared `L`, inspired
by Anderson--Bjarnadottir but **not** a reproduction of their threshold/link
selection. On a second seed (`20260923`, five draws per design, `n=200`),
median absolute point-estimation errors were **0.0203** for the LP, **0.0335**
for external kNN, and **0.1010** for Pythia's pair center. The LP was below the
known MAE floor in 44/50 draws, illustrating why its point result is not a
95% lower confidence bound. The valid paired lower bound was below truth in
50/50, but at this small sample it is substantially more conservative.
This is not a Pythia estimator win; it is a necessary comparator and shows
that the model-free estimator remains the active research gap.
An outcome-blind half-sample jackknife of the LP, still a train-free *point*
estimator without a lower-bound guarantee, reduced second-seed median error
to **0.0145** (versus LP **0.0203** and external kNN **0.0335**) at `n=200`.
The first seed was less favorable: jackknife **0.0220**, kNN **0.0214**.
High-dimensional, heavy-tail, and interaction designs remained counterexamples;
this preliminary bias correction cannot be advertised as generally superior.
On a larger untouched seed (`20260924`, 20 draws per design, `n=200`), median
absolute errors were jackknife **0.0200**, external kNN **0.0275**, raw LP
**0.0293**, and paired center **0.1081**. Yet kNN was the unique closest of
these four in 88/200 draws, jackknife in 59/200, and kNN had lower median
error in four of the ten design families. Median aggregate improvement is
real for this fixed benchmark but not a universal estimator win.

## Mayo reconstruction and multi-loss results

`tests/test_guarded.py::test_frozen_mayo_arm_b_reproduces_exactly` recomputes
the historical Arm B artifact from private frozen OOF predictions:

| Quantity | Minutes | Tier |
|---|---:|---|
| Arm B stratified guarded MAE bound | 28.8495996043 | calibrated |
| global floor estimator | 31.1186421201 | calibrated |
| best audited OOF MAE | 32.8700117033 | retrospective achieved error |
| Qcap conservative rung | 24.1867259409 | conditional on guard assumptions |

The two-point theorem **does not reconstruct or prove 28.8496**: that number
comes from the legacy two-dimensional simulated guard, with a different
calibration basis and unit of analysis. Reproduction is not certification.

Run `python3 scripts/run_mayo_decision_losses.py` for an aggregate-only audit
of the earliest case per patient in four chronological panels (2,632 cases
each), using frozen OOF predictions and two existing feature metrics. At
illustrative `L=300`, the tabular-metric **grouped** pinball-0.9 lower bounds
were **2.187, 2.450, 2.053, 1.645** minute-equivalent pinball units, versus
selection-adjusted paired bounds **1.886, 2.033, 1.730, 1.598**. Achieved
frozen OOF pinball losses were 19.99, 20.32, 18.09, and 17.32. The tabular
**grouped W30 miss-risk** lower bounds were **0.201, 0.202, 0.181, 0.180**,
versus paired **0.138, 0.150, 0.126, 0.115**. Achieved miss fractions were
0.392, 0.385, 0.363, and 0.356. Across the four panels and both feature
metrics, grouped exceeded paired in 15/16 loss-representation-period cells.
These bounds are still weak, though nonzero, and all remain
**exploratory**: `L=300`, the 900-minute pinball outcome-difference cap,
smoothness of the selected quantile/coverage center, calendar-patient
independence, and pre-outcome provenance are not independently justified.
The full-cohort OOF winner was selected retrospectively. They do not imply
the note representation has no information or that its bound must exceed
the tabular bound.

## Literature and release gates

- Blackwell, [Equivalent Comparisons of Experiments (1953)](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full): pre-existing decision-value order.
- Maurer and Pontil, [Empirical Bernstein Bounds and Sample Variance Penalization (2009)](https://arxiv.org/abs/0907.3740): concentration tool, not a new Pythia inequality.
- Anderson and Bjarnadottir, [As good as it gets? A new approach to estimating possible prediction performance (2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296904): particularly close no-predictor-training constrained-omniscient MAE floor heuristic. It solves a global label-aware smoothness-constrained optimization, and the authors state it has no proven performance guarantee. The grouped witness is a different finite-design, uncertainty-adjusted construction, but no superiority claim is justified until this LP is reimplemented under the same metrics and DGPs.
- Müller, Luo, and Barber, [Are all models wrong? Fundamental limits in distribution-free empirical model falsification (2025)](https://proceedings.mlr.press/v291/muller25a.html): a different estimand; their impossibility results caution against broad distribution-free claims.

Before a released multi-loss certificate: externally establish center
smoothness and feature timing, account for shared-patient/period dependence,
freeze metric and clip/design grids, validate caps, apply simultaneous error
allocation, and audit a held-out cohort. A paper novelty claim further needs
independent proof and a complete related-work comparison. If these gates fail,
the API must downgrade or refuse.
