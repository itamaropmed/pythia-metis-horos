# Curvature-corrected SAA floor certificates

## Scope and claim

This note states the new conditional Pythia route precisely.  The elementary
smoothness inequality is classical; it is **not** claimed as a new inequality.
The research contribution under evaluation is its composition with
outcome-blind block SAA, simultaneous one-sided concentration, independent
declarations, and fail-closed certificate provenance for attainable-risk
floors.

The route is training-free on the certification sample.  An independently
learned blocking key is allowed, but its fitting belongs to representation
construction and must be completed before certification outcomes are opened.

## Theorem

In one frozen block let unit `j` have law `Q_j`, scalar expected loss

`L_j(a) = E_{Y~Q_j} loss(Y,a)`, Bayes action `a_j*`, and Bayes risk
`R_j = L_j(a_j*)`.  Define the empirical SAA witness

`S = inf_a mean_j loss(Y_j,a)`.

No within-block independence is needed for the expectation step:

`E S <= inf_a mean_j L_j(a)`.

Suppose `L_j` is differentiable at its minimizer and has curvature at most
`kappa_j` on the segment joining `a_j*` to a common action `c`.  The descent
lemma gives

`L_j(c) <= R_j + (kappa_j/2)(c-a_j*)^2`.

Therefore

`E S <= mean_j R_j + inf_c mean_j (kappa_j/2)(c-a_j*)^2`.

The infimum is attained at the curvature-weighted mean

`c_kappa = sum_j kappa_j a_j* / sum_j kappa_j`.

For a common curvature bound `kappa`, this reduces to

`E S <= mean_j R_j + (kappa/2) Var_j(a_j*)`.

For an action-Lipschitz loss, the independent first-order inequality

`E S <= mean_j R_j + C inf_c mean_j |c-a_j*|`

also holds.  Its centre is a median.  The minimum of the two penalties is
valid because both upper-bound the same Jensen gap.

For standard scaling:

- MAE has curvature `2 sup_y f_j(y)` and action Lipschitz constant 1;
- pinball at level `q` has curvature `sup_y f_j(y)` and action Lipschitz
  constant `max(q,1-q)`;
- MSE has curvature exactly 2, making its quadratic penalty exact.

## What is and is not observed

The certified outcomes identify the block witness, not the within-block Bayes
actions.  Thus neither the action dispersion nor a density/curvature envelope
is recoverable without additional structure from one outcome per unit.  Each
must be:

1. externally declared;
2. calibrated on honest repeated environments; or
3. reported as a sensitivity axis.

The public API returns the last case as `conditional` / `sensitivity_only`.
It refuses missing provenance and refuses a local curvature declaration when
the declared action displacement leaves its registered radius.

## Novelty boundary

Variance bounds on Jensen gaps are established mathematical tools; see Gao,
Sitharam and Roitberg, *Bounds on the Jensen Gap* (2019),
https://arxiv.org/abs/1712.05267.  Classical information-theoretic Bayes-risk
lower bounds such as Chen, Guntuboyina and Zhang (JMLR 2016),
https://jmlr.org/papers/v17/16-185.html, address a different decision-theoretic
setting.  Anderson and Bjarnadottir (PLOS ONE 2024),
https://doi.org/10.1371/journal.pone.0296904, target dataset-level attainable
MAE but explicitly describe their LP procedure as heuristic without coverage
guarantees.

Accordingly, the defensible novelty candidate is not “we discovered a Jensen
gap bound.”  It is a finite-sample certificate architecture for dataset-level
attainable risk that (a) uses SAA witnesses without fitting a target predictor,
(b) exposes non-identifiable geometry as a declared sensitivity object, (c)
uses loss curvature to make that sensitivity second order, and (d) composes
the evidence under an explicit alpha/provenance ledger.

This remains a candidate until an independent proof audit and a matched
implementation of the Anderson--Bjarnadottir protocol are complete.
