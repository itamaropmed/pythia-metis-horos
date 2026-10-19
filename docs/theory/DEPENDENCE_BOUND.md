# Dependence-capped squared-loss bound

Status: proved algebraically at the population level; finite-sample version is
conditional on the explicitly stated bounded-i.i.d. evaluation contract.

## Setup

Let `m(X) = E[Y|X]`, `epsilon = Y-m(X)`, and let frozen predictor `i` satisfy
`f_i(X)=m(X)+e_i(X)`. Under squared loss,

```text
r_i = E[(Y-f_i)^2] = R* + a_i,
a_i = E[e_i^2],
R* = E[epsilon^2].
```

For each pair,

```text
d_ij = E[(f_i-f_j)^2] = a_i+a_j-2c_ij,
c_ij = E[e_i e_j].
```

Assume finite second moments and a declared cap

```text
c_ij <= lambda sqrt(a_i a_j),  0 <= lambda < 1.
```

Only positive shared error is capped; negative covariance makes disagreement
larger and the resulting lower bound more conservative.

## Theorem 1: sharp two-predictor population bound

For a pair `i,j`, put

```text
s = a_i+a_j,
delta = a_i-a_j = r_i-r_j,
d = d_ij.
```

Since `2 sqrt(a_i a_j)=sqrt(s^2-delta^2)`, the correlation assumption implies

```text
s-d <= lambda sqrt(s^2-delta^2).
```

The largest compatible excess-risk sum is

```text
s_max = [d + lambda sqrt(d^2-(1-lambda^2)delta^2)]/(1-lambda^2).
```

At `lambda=0`, this reduces continuously to `s_max=d`. Therefore

```text
R* = (r_i+r_j-s)/2
   >= (r_i+r_j-s_max)/2.
```

Taking the maximum over all pairs remains a valid lower bound because every
pairwise inequality holds simultaneously. A negative radicand or
`s_max < |delta|` means the supplied population moments are incompatible with
the correlation cap, so the implementation refuses.

The bound is exact for a pair when its positive correlation reaches the cap.
Increasing `lambda` expands the feasible set and cannot increase the bound.

## Theorem 2: finite-sample bounded-i.i.d. rung

For a simpler robust construction, use

```text
2c_ij <= 2 lambda sqrt(a_i a_j) <= lambda(a_i+a_j),
d_ij >= (1-lambda)(a_i+a_j).
```

Thus

```text
R* >= 0.5 [r_i+r_j-d_ij/(1-lambda)].
```

Assume fixed predictors are evaluated on `n` independent identically
distributed rows and `|Y-f_i(X)| <= A` almost surely. Squared risks lie in
`[0,A^2]`, while squared disagreements lie in `[0,4A^2]`. Construct
simultaneous one-sided lower confidence bounds `r_i^-` and upper confidence
bounds `d_ij^+`. On their joint coverage event,

```text
B_ij = 0.5 [r_i^- + r_j^- - d_ij^+/(1-lambda)] <= R*.
B = max(0, max_ij B_ij).
```

The implementation offers Hoeffding concentration and a sharper empirical
Bernstein option using Maurer and Pontil's sample-variance bound. Failure
probability is divided across all risk and disagreement moments by a union
bound. The maximum over pairs is safe because all component inequalities hold
on the same simultaneous event.

This theorem does **not** automatically cover ordinary cross-fitted rows:
training-set overlap can make their prediction errors dependent across rows.
For OOF use, Pythia must add an entity/fold-aware concentration theorem or label
the result calibrated. The present proved finite-sample rung is appropriate for
a frozen predictor pool scored on an untouched independent evaluation set.

## Estimation

Plugging empirical moments into Theorem 1 produces a high-accuracy floor
estimator in the current simulations, but it is not itself a finite-sample
lower confidence bound. Pythia reports it as an estimator/sensitivity value. The
bounded empirical-Bernstein construction is the lower confidence bound.

## Theorem 3: sharp bounded-range patient rung

Let the independent sampling unit be a patient and let each patient's moment
be its mean over that patient's cases. If patients are iid, `|Y-f_i| <= A`,
and `|f_i-f_j| <= D` almost surely, patient risk means lie in `[0,A^2]`
and patient disagreement means in `[0,D^2]`. Applying the two-sided empirical-
Bernstein inequality with a union bound gives simultaneous rectangles

```text
r_i in [r_i^-, r_i^+],   d_ij <= d_ij^+.
```

Instead of substituting these into Theorem 2, Pythia minimizes the exact
Theorem-1 expression over every compatible rectangle. Parameterize a pair by
`delta=r_i-r_j`. For fixed `delta`, the smallest feasible risk sum is

```text
s_min(delta) = max(2 r_i^- - delta, 2 r_j^- + delta).
```

The exact lower-bound objective is convex on each side of the kink
`delta=r_i^- - r_j^-`. Its only possible interior stationary points are
`delta=+d_ij^+` and `delta=-d_ij^+`; consequently evaluating the interval
endpoints, kink, and feasible stationary points gives the global minimum
without numerical optimization. Taking the larger of this sharp rectangle
bound and Theorem 2 is valid because both hold on the same simultaneous event.

This rung removes Catoni's predeclared variance caps. It does not remove the
correlation cap, the honest range requirements, or the need for predictors
frozen before the independent evaluation patients arrive. Its estimand is
equal-patient expected mean squared case error.

## Empirical Bernstein source

Andreas Maurer and Massimiliano Pontil, “Empirical Bernstein Bounds and Sample
Variance Penalization,” COLT 2009: https://arxiv.org/abs/0907.3740

## Heavy-tail patient-level extension

`catoni_cluster_pairwise_bound` first aggregates each moment within patient,
then applies Catoni's finite-variance M-estimator across independent patients.
It allocates failure probability simultaneously across every predictor risk and
pairwise disagreement. This removes the almost-sure loss cap, but requires
predeclared upper bounds on patient-level variances and changes the estimand to
equal-patient expected mean case loss. See Catoni's finite-variance analysis:
https://arxiv.org/abs/0909.5366.
