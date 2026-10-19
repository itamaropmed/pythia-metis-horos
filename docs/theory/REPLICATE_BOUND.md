# Outcome-blind near-replicate bound

Status: algebraically proved for the declared paired estimand; finite-sample
confidence is conditional on iid randomized blocks, outcome-blind metric and
pair construction, distinct sampling entities, honest range caps, and an
externally justified Lipschitz cap. It is not yet an independently audited
paper theorem.

There is also a fixed-design rung. It conditions on the complete audited
covariate set and outcome-blind matching, assumes conditionally independent
entity outcomes, and targets average Bayes risk over those exact endpoints.
It does not require iid covariates or randomized blocks and must not be
reported as a sampling-population certificate.

## Why this rung exists

Predictor disagreement cannot detect bias shared by every predictor. No method
using only those predictions and outcomes can remove that non-identifiability
without another assumption or source of information. Near-replicate outcomes
provide a second information source: outcome variation between cases at the
same or nearby information state.

Pythia still trains no model. Horos constructs and freezes pairs from a declared
representation; Pythia consumes only paired outcomes, distances, entity IDs,
block IDs, and a versioned guard.

## MAE theorem

For covariates `x_i,x_j`, choose conditional medians `m_i,m_j`. The triangle
inequality gives

```text
|Y_i-Y_j| <= |Y_i-m_i| + |Y_j-m_j| + |m_i-m_j|.
```

If the selected median is `L`-Lipschitz under the frozen metric,
`|m_i-m_j| <= L d_ij`, then

```text
0.5 E[|Y_i-Y_j| - L d_ij | x_i,x_j]
    <= 0.5 (R_i^MAE + R_j^MAE).
```

At exact replicates the correction is zero. The result does not assume
symmetric, Gaussian, or homoskedastic noise and is unaffected by common bias in
a predictor pool.

## MSE theorem

For conditional means `mu_i,mu_j` and conditionally independent sampling
entities,

```text
E[(Y_i-Y_j)^2 | x_i,x_j]
    = Var(Y_i|x_i) + Var(Y_j|x_j) + (mu_i-mu_j)^2.
```

If `|mu_i-mu_j| <= L d_ij`, then

```text
0.5 E[(Y_i-Y_j)^2 - (L d_ij)^2 | x_i,x_j]
    <= 0.5 (R_i^MSE + R_j^MSE).
```

## Finite-sample design

Independent entities are assigned to fixed-size randomized blocks without
outcomes. A fixed covariate-only algorithm forms disjoint pairs inside every
block. All pair scores in a block are averaged, and concentration is applied
only across iid blocks. This avoids the false claim that globally matched pair
statistics are independent.

For bounded outcomes and distances, the block score has a known range. Pythia
offers empirical Bernstein and Hoeffding lower confidence bounds, plus a
Bonferroni-simultaneous Lipschitz sensitivity curve. The public
API refuses unless the pair artifact explicitly records an external audit of
outcome blindness and iid block sampling; the generic constructor deliberately
does not self-certify those facts.

For a frozen fixed design, the smoothness penalty is deterministic and
disjoint-pair outcome terms are independent, though generally not identically
distributed. Pythia therefore offers two pair-level conditional bounds:

- one-sided Hoeffding over every pair; and
- variance-adaptive empirical Bernstein using Maurer and Pontil (2009),
  Theorem 11, which explicitly permits independent non-identical bounded
  variables.

Both subtract the observed smoothness penalty exactly and charge sampling
uncertainty only to the outcome term. Horos can use a deterministic repeated
nearest-neighbor construction over the full frozen representation. This is
valid for the conditional rung because matching is fixed before outcomes and
every entity appears in exactly one pair. The constructor records facts it can
verify but leaves outcome blindness and conditional entity independence false
until an external provenance audit attests them.

When entity outcomes may be dependent within known operational clusters,
Pythia provides conditional block Hoeffding and conditional block empirical
Bernstein. They average outcomes inside each declared block, permit arbitrary
within-block dependence, and require independent (not necessarily identical)
blocks. Unequal block sizes deliberately target an equal-block-weighted risk;
they are never silently presented as entity-weighted.

The public guard accepts `concentration="auto"`. It chooses pair-level
conditional empirical Bernstein only with an audited entity-independence flag,
downgrades to conditional block empirical Bernstein with audited independent
blocks, and otherwise returns a structured refusal. This makes the dependence
decision automatable without converting an unverified assumption into a
certificate.

The same outcome-term event covers every Lipschitz value because changing `L`
only changes a deterministic correction. Conditional sensitivity curves
therefore incur no multiplicity penalty over the `L` grid (indeed, the event
covers the continuum); an ensemble still Bonferroni-corrects over alternative
pair designs. Population block curves retain their grid correction because
their random block scores and ranges depend on `L`.

The estimand is the expected block-average Bayes risk of pair endpoints under
the declared pairing design. When every entity is used exactly once in equal-
size blocks, this equals the sampling-population average risk. Thresholding or
discarding pairs changes the estimand to the explicitly retained population.

## Falsification evidence

In 3,600 simulations spanning MAE/MSE, uniform and spike-mixture noise, slopes
0/1/3, and 1,000/4,000/10,000 entities, the correctly adjusted empirical-
Bernstein rung had 3,600/3,600 population-level valid results (95% Wilson lower
0.9989). Median usable tightness was 0.466 versus 0.118 for matched Hoeffding.
Empirical Bernstein was strictly tighter in 2,853/3,600 runs.

The negative controls behave as required: using no smoothness correction was
valid in only 78.7% of runs, and using half the true Lipschitz constant in
88.3%. Under the high-slope design those rates fell to 52.8% and 66.8%. This
shows that the smoothness cap is a substantive assumption, not decorative
metadata.

A broader 1,500-run adversarial suite added nonlinear, heteroskedastic,
asymmetric, and five-dimensional designs. The corrected population rung was
valid 1,500/1,500 (Wilson lower 0.9974); the uncorrected method covered only
77.1%. It also exposed the expected curse of dimensionality: randomized-block
matching often returned a zero bound. Global outcome-blind pairing reduced
median pair distance to 0.65% of the blocked distance and raised median
fixed-design Hoeffding tightness from 0 to 0.352, with 1,500/1,500 validity.
Conditional empirical Bernstein was valid in all 1,500 runs (Wilson lower
0.9974) and raised median tightness from 0.250 with blocked matching to 0.624
with global matching. It improved on blocked conditional empirical Bernstein
in 1,356/1,500 runs. At 10,000 entities median tightness reached 0.752; it was
0.597 for MAE and 0.721 for MSE. These comparisons share the fixed-design
estimand; they are not comparisons against the population-risk rung.

In a separate 500-run MSE stress test, 4,000 pairs shared strong latent scale
shocks within 200 independent blocks. The invalid shortcut that treated pairs
as independent covered only 467/500 (93.4%; Wilson lower 0.909). Conditional
block empirical Bernstein covered 500/500 (Wilson lower 0.992), with median
tightness 0.272. The deliberately lower tightness is the price of honest
within-cluster dependence, not a regression to optimize away.

## Mayo diagnostic

The retrospective diagnostic uses the earliest scheduled case for each of
10,532 patients, fixed domain scaling, random blocks of eight, and greedy mean-
L1 pairing. It uses 10,528 patients in 1,316 blocks. The best existing OOF MAE
on these rows is 32.8284 minutes.

The table below uses a predeclared ten-seed registry and Bonferroni allocation
over all ten designs and nine Lipschitz values. It is therefore safe to select
the largest row-wise bound after observing the outcomes.

| Lipschitz cap | tabular floor | tabular + note value/applicability floor |
|---:|---:|---:|
| 0 | 39.40 | 38.53 |
| 100 | 36.12 | 33.96 |
| 200 | 32.84 | 29.40 |
| 300 | 29.56 | 24.83 |
| 400 | 26.28 | 20.26 |
| 500 | 23.00 | 15.68 |
| 600 | 19.71 | 11.11 |

The new global fixed-design empirical-Bernstein diagnostic is materially less
sensitive to the block geometry. Its tabular curve runs from 26.61 minutes at
`L=0` to 23.78 at `L=600`; the note-value/applicability curve runs from 28.41
to 22.50. Its single-event simultaneous radius is 1.89--1.92 minutes rather
than roughly 10 minutes under the blocked fixed-design Hoeffding ensemble.
These are still exploratory: the
Lipschitz cap, outcome cap, outcome-blind feature provenance, and conditional
independence of patient outcomes have not been externally audited. The two
representations also define different conditional estimands, so their floor
difference is not by itself a certified value-of-notes comparison.

As a dependence stress test, Mayo cases were also paired only within ISO
calendar weeks and inference was performed across 170 week blocks. The
outcome-blind rule retained 10,444 patients and dropped 84 deterministically
from odd-sized weeks. The tabular diagnostic falls from 12.50 minutes at
`L=0` to 2.99 at `L=600`; the note-value diagnostic falls from 12.92 to zero.
This is the appropriately wider real-world downgrade, but it is still
exploratory: calendar weeks are a plausible clustering unit, not proof that
different weeks are independent. Site/team and serial dependence require a
prospective design or a stronger dependence theorem.

Values above the achieved OOF loss do not disprove the algebra; they falsify
the corresponding too-small Lipschitz cap or another sampling/design
assumption for this retrospective use. No row in this curve is selected as a
certificate. The entire displayed grid has simultaneous 95% coverage under
its declared assumptions. The ten-seed registry was less efficient than the
single frozen seed because its multiplicity penalty exceeded the pairing gain;
it is retained as a negative control and safe optional mechanism, not selected
as the default on efficiency grounds. The result identifies the required next experiment: freeze the
metric and smoothness audit on historical data, then apply the pair design to
untouched prospective patients.

## Relation to prior work

Graph and nearest-neighbor variance estimators are established. Madrid Padilla
(JMLR 2024) studies consistent homoskedastic and heteroskedastic variance
estimation on general and nearest-neighbor graphs. Guyader and Hengartner
(JMLR 2013) prove consistency and optimal rates for mutual-nearest-neighbor
regression, while Ye and Madrid Padilla (JMLR 2021) develop nonparametric
quantile regression on nearest-neighbor graphs. Pythia's intended distinction is
not nearest-neighbor prediction: it is a fail-closed, model-free finite-sample
lower certificate for attainable MAE/MSE using disjoint randomized-block pairs
and explicit smoothness sensitivity. No novelty claim is made until an
independent proof and literature audit are complete.

- https://jmlr.org/papers/v25/23-1061.html
- https://jmlr.org/papers/v14/guyader13a.html
- https://jmlr.org/papers/v22/20-1462.html
- https://arxiv.org/abs/0907.3740
