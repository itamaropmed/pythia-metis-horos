# Fixed-design replicate theorem audit

Status: internal proof audit; conditional theorem implemented and tested, not
yet independently reviewed for publication.

## Statement and estimand

Freeze covariates `x_1,...,x_2m` and an outcome-blind perfect matching `M`.
Every sampling entity occurs in exactly one pair. Let `r_i` be the conditional
Bayes risk at `x_i`, and define the audited-covariate target

```text
R_M = (1 / 2m) sum_i r_i.
```

For pair `k=(i,j)`, define outcome term `A_k` and deterministic correction
`C_k(L)`:

Without any center-movement constraint, a positive finite-design lower bound
is impossible in general: when all audited covariates are distinct, the
deterministic conditional law `Y=m(X)` with `m(x_i)=y_i` fits the observed
outcomes exactly and has Bayes MAE/MSE zero. A noisy law can also produce the
same observations. Thus the observed outcomes alone cannot certify positive
irreducible risk. The metric/smoothness and sampling contracts supply the
identifying restrictions; they are not optional numerical tuning knobs.

| loss | `A_k` | `C_k(L)` | center condition |
|---|---:|---:|---|
| MAE | `|Y_i-Y_j|/2` | `L d_ij/2` | a selected median is `L`-Lipschitz |
| MSE | `(Y_i-Y_j)^2/2` | `(L d_ij)^2/2` | the conditional mean is `L`-Lipschitz |

Then

```text
R_M >= (1/m) sum_k { E[A_k | x] - C_k(L) }.
```

For MAE this follows from the triangle inequality around the selected
conditional medians. It needs no distributional symmetry. For MSE, expansion
around the two conditional means gives the result when distinct entity
outcomes are conditionally uncorrelated; the implementation declares the
stronger, easier-to-audit condition of conditional independence.

This is a fixed-design target. It is not the Bayes risk under a future
covariate distribution unless an additional sampling/transport argument is
supplied.

## Finite-sample lower confidence bound

Assume the pair outcome terms are conditionally independent and bounded in
`[0,U]`, where `U=D/2` for MAE and `U=D^2/2` for MSE. The terms need not be
identically distributed. Let `Abar` be their sample mean and `S^2` their
unbiased sample variance. Applying the lower-tail form of Maurer--Pontil
Theorem 11 to `A_k/U` gives, with probability at least `1-delta`,

```text
mean_k E[A_k | x]
  >= Abar
     - sqrt(2 S^2 log(2/delta) / m)
     - 7 U log(2/delta) / (3(m-1)).
```

Subtracting the observed mean correction and truncating at zero yields the
implemented lower bound. One-sided Hoeffding is retained as a variance-free
fallback. The deterministic correction is not included in the concentration
range or variance.

Primary concentration source: Maurer and Pontil, *Empirical Bernstein Bounds
and Sample Variance Penalization* (2009), Theorem 11:
https://arxiv.org/abs/0907.3740.

## Simultaneous smoothness curve

The random quantity `Abar - radius` is identical for every `L`; only the
deterministic `Cbar(L)` changes. Therefore the same probability event implies
the bound for every `L >= 0` simultaneously. No union bound over a displayed
grid is required. If Pythia selects among `K` predeclared outcome-blind pair
designs, it uses a union bound over `K`, but still not over `L`.

## Why the MAE corrected center is not a universally accurate floor estimate

The bound uses only a pairwise absolute-difference statistic. That statistic
does not identify Bayes MAE. At an identical covariate value, consider two
different conditional noise laws:

| noise law | `E|Y-Y'|/2` | Bayes MAE |
|---|---:|---:|
| `Y=±1` with equal probability | 0.5 | 1.0 |
| `Y~Uniform[-1.5,1.5]` | 0.5 | 0.75 |

Both have the same population pair statistic but different Bayes MAE. Thus
the clipped corrected center in the API is a *lower-bound plug-in*, not a
distribution-free point estimator of Bayes MAE. It is clipped at zero so
`0 <= bound <= estimate` even when the smoothness penalty exceeds the pair
term. This counterexample concerns the pair statistic, not all information in
the full conditional outcome distribution. A separate shape assumption or
calibrated estimator is required for a defensible MAE floor point estimate.
The existing Mayo 31.1186-minute `E2` estimate remains calibrated, not proved.

## Metric geometry and sparse features

The theorem needs a valid *bound on conditional-center movement*, not a
particular embedding or equal-weight feature distance. If an independently
justified coordinatewise modulus satisfies

```text
|m(x)-m(x')| <= sum_j ell_j |x_j-x'_j|,
```

that right-hand side can replace `L d(x,x')` pair by pair. Equivalently,
declare it as the distance with `L=1`. This may be much sharper than a
single global `L` times mean-L1 in a sparse setting. The weights must be
fixed without reading the evaluation outcomes, and the matching must be
constructed from permitted pre-outcome features. Discovering useful weights
from outcomes belongs in a separate, honest training/development stage and
cannot be smuggled into Pythia inference. A known-support synthetic control
tests this distinction; it is not evidence that a signal-aligned metric can
be discovered automatically on Mayo.

Information and bound geometry must not be conflated. If `Z=f(X)` is a
deterministic derived feature, `sigma(X,Z)=sigma(X)`, so the population Bayes
risk is exactly unchanged for either loss. If `Z` is a randomized garbling
conditionally independent of `Y` given `X`, the Bayes risk is likewise
unchanged when `X` remains available. Yet a finite-sample near-replicate
bound may move when the metric or matching changes; this is not a change in
the information content. Conversely, adding `Y` itself as a feature makes
Bayes risk zero, but is post-outcome leakage for a deployable task. If an
oracle mask reveals `Y` independently with probability `p`, the augmented
Bayes risk is `(1-p)` times the original Bayes risk; for outcome-dependent
masking that simple identity need not hold.

## Cluster-dependent downgrade

If outcomes may be dependent within declared blocks, replace pair terms by
block-average outcome terms. Maurer--Pontil applies when the block vectors are
independent; their distributions may differ and dependence inside a block is
unrestricted. The resulting target weights blocks equally. It equals the
entity-weighted target only when blocks contain the same number of pair
endpoints. `concentration="auto"` selects this rung when block independence,
but not entity independence, is audited.

## Fail-closed conditions

Pythia refuses or remains exploratory when any required claim is absent:

- pairing or representation construction could see outcomes;
- an entity appears in multiple pairs;
- neither entity nor block independence is audited;
- the declared outcome-difference cap is violated;
- no external justification supports the smoothness cap;
- feature timing permits post-outcome or post-decision leakage; or
- a fixed-design result is requested as a deployment-population certificate.

For clinical notes, embedding quality is not itself an assumption certificate.
The representation hash, time cutoff, entity isolation, distance definition,
and smoothness argument must be audited independently of observed case
durations.

## Evidence and unresolved issues

The global conditional empirical-Bernstein method was valid in 1,500/1,500
adversarial simulations (Wilson lower 0.9974), with median tightness 0.624.
A cluster-shock test showed why the downgrade is necessary: the false
pair-independence shortcut covered 93.4%, while block inference covered
500/500. These are falsification results, not a proof substitute.

Independent review must still check the loss inequalities, scaling constants,
conditioning argument, matching measurability, and relation to graph-based
variance estimation. No novelty claim is authorized by this document.
