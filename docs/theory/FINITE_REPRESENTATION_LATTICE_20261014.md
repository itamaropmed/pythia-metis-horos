# Finite representation-lattice certificates

## Problem

A clinical information study rarely contains one representation.  It contains
a graph: global population, procedure family, procedure, note concept, and a
possibly fragmented text neighborhood.  Inspecting all nodes and reporting the
largest lower endpoint is invalid unless the stochastic events are
simultaneous.  Treating every representation independently also throws away
two structural facts:

1. a deterministic garbling cannot lower Bayes risk; and
2. a bijection or conditionally independent nuisance channel leaves Bayes risk
   exactly unchanged.

The finite representation lattice composes these facts with the finite-cell
SAA endpoint.  It does not train a predictor.

## Conditional theorem

Let `V` be a finite registry frozen before outcomes.  Node `v` exposes a
finite label `Z_v`.  For every directed edge `v -> u`, assume
`Z_u = g_vu(Z_v)` under the population law and common decision timing.  For
every registered equality pair assume either a measurable bijection or
`N independent of Y conditional on Z`, so the two nodes have equal Bayes
risk for the declared loss and action space.

Conditional on the audited labels, define

```text
S_v = (1/n) sum_z min_a sum_{i: Z_vi=z} loss(Y_i, a).
```

For each cell let changing one outcome alter its optimal empirical loss by at
most the externally declared hard loss range `D_vz`.  Then

```text
L_v = max(0, S_v - sqrt(log(1/alpha_v)/2
                        * sum_i (D_v,Z_vi / n)^2))
```

is a lower endpoint for the Bayes risk at node `v`, targeting the audited
empirical mixture of cell probabilities.  The result follows from the
downward bias of sample-average minimization and conditional bounded
differences.  If `sum_v alpha_v <= alpha`, all node statements hold
simultaneously with probability at least `1-alpha`.

On that simultaneous event:

- if `v -> u` is a garbling, `R_v <= R_u`, so every lower endpoint for `v` is
  also a lower endpoint for `u`;
- if `v` and `u` are a registered equality pair, the maximum of their lower
  endpoints lower-bounds their common risk;
- selecting the strongest endpoint after simultaneous closure costs no
  additional alpha.

The implementation additionally checks that each observed rich label maps to
only one poor label.  This can falsify a claimed deterministic map but cannot
prove its population or timing validity.  Missing attestations therefore
produce a refusal.

## What is new and what is not

The Blackwell order and its Bayes-risk implication are classical.  Modern
information-risk bridges give a much broader geometry of experiments and
risks.  Finite-cell empirical optimization and bounded-difference
concentration are also classical ingredients.  The candidate Pythia
contribution is narrower:

- a training-free, finite-sample lower-risk endpoint at every node;
- a fail-closed representation/provenance contract;
- one familywise alpha ledger over a predeclared information DAG;
- equality-class repair of information-preserving text fragmentation; and
- order-constrained diagnostic estimates separated from certified lower
  endpoints.

For an attested equal-risk class, the diagnostic estimator is routed to the
member with the fewest observed cells, with frozen registry order as a tie
break.  Refining an empirical decision table can only reduce its empirical
optimum.  Because the population optimum is equal across the class, the
least-fragmented score has weakly smaller downward bias in expectation.  This
does not imply universal absolute-error or squared-error dominance.  A fresh
ten-mechanism confirmation nevertheless improved median absolute error from
0.1345 to 0.0130 and won 496/500 paired runs.

This composition should not be advertised as a new information-processing
theorem until an independent proof and priority audit are complete.

## Relation to modern literature

- Williamson and Cranko's
  [Information Processing Equalities and the Information--Risk Bridge](https://www.jmlr.org/papers/v25/22-0988.html)
  provides a geometric bridge between information measures and Bayes risk.
- Mémoli, Vose and Williamson's
  [Risk distance for supervised learning problems](https://jmlr.org/papers/volume26/24-0322/24-0322.pdf)
  formalizes comparison of learning problems and explicitly relates it to
  Blackwell experiments.
- Esposito, Vandenbroucque and Gastpar's
  [Lower Bounds on the Bayesian Risk via Information Measures](https://www.jmlr.org/papers/v25/23-0361.html)
  derives estimator-independent impossibility bounds through information
  measures and data processing.  Its estimand and evidence route differ from
  the empirical finite-representation floor here.
- Antos et al.'s classical no-rate result for unrestricted Bayes-error
  estimation explains why Pythia must declare finite representations, support,
  sampling and timing contracts rather than promise a universal
  distribution-free estimator.
- Hierarchical stratification and selective-inference work addresses adaptive
  partition choice, but does not by itself supply this Bayes-floor lattice.

## Remaining theorem gaps

1. Replace iid rows by patient-clustered and temporally stable events before a
   Mayo certificate.
2. Permit randomized garblings with independently audited Markov kernels.
3. Develop sharper alpha allocation or one joint supermartingale without
   choosing it on certification outcomes.
4. Pair lower endpoints with externally frozen policy-risk upper endpoints to
   certify a positive value of added information, not merely its floor.
5. Formalize support propagation: a coarse cell range must cover the union of
   every descendant conditional support.
6. Prove or reject an anytime-valid version for Metis' sequential proposal
   loop.
7. Compare the equality-routed point estimator with published Bayes-floor
   estimators outside deliberately fragmented finite representations.

## Conditional refinement signal test

The lattice is now accompanied by a separate, model-free falsification test.
For a frozen coarse label `G` and refinement `Z`, its statistic is the empirical
SAA reduction from `G` to `(G,Z)`.  It permutes `Z` within `G` and uses a
plus-one Monte Carlo tail probability.  Under exact conditional exchangeability
and `Y independent of Z given G`, this is a finite-sample conditional
randomization test for any registered loss statistic.  It is related to the
conditional-randomization and conditional-permutation literature, but differs
from model-based permutation importance because its statistic is the empirical
Bayes-action optimum and no predictor is fit.

The conditional-independence null is known to be impossible to test
distribution-free without structure in general.  Here, finite frozen strata
and within-stratum exchangeability are the explicit structure.  Missing
exchangeability produces only an exploratory tail area.  Even a valid
rejection establishes signal, not a lower confidence bound on the size of the
population Bayes-risk reduction.

Relevant context includes the model-X conditional randomization framework and
[conditional permutation importance](https://arxiv.org/abs/2309.07593), as
well as analyses of exact versus approximate conditional permutation schemes.
The Pythia contribution candidate is the training-free SAA decision statistic,
its integration with the representation lattice, and the same fail-closed
timing/provenance boundary used by the risk certificates.

## Magnitude bridge through a frozen refined policy

A conditional-independence rejection does not quantify information value.  A
separate bridge now does so when an externally frozen policy exists.  Let
`R_C` and `R_F` be coarse and refined Bayes risks and let `pi` be measurable
using only the refined information.  Because `R_F <= R(pi)`, simultaneous
endpoints obey

```text
R_C - R_F >= R_C - R(pi) >= L_C - U_pi.
```

Pythia computes `L_C` from finite-cell SAA and `U_pi` from bounded independent
policy losses.  It supports predeclared Hoeffding, empirical-Bernstein,
Bentkus and dKelly upper-mean routes; selecting the smallest after outcomes is
allowed only when all candidates were constructed simultaneously and charged
to the alpha ledger.  It refuses a policy selected on evaluation outcomes,
post-outcome inputs, or a policy not measurable from the declared refined
representation.

This bridge is elementary decision theory rather than a new inequality.  Its
candidate contribution is operational: a model-agnostic, fail-closed path
from a Bayes-floor lower witness to a positive lower confidence bound on the
value of a representation, with model fitting kept outside Pythia.  On 500
fresh known-law runs it had zero violations, was positive in every run and
captured a median 64.3% of true information value versus 45.5% for the matched
global-range/Hoeffding baseline.

## Software

`pythia.core.finite_cell_lattice.finite_cell_information_lattice` implements
the registered finite deterministic case.  It refuses unattested nesting,
unattested equality, observed non-nesting, target-derived labels, support
violations and any refused node endpoint.
