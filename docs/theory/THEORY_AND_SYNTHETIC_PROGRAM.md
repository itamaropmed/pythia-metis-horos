# Pythia theory and synthetic validation program

Status: research specification and first exploratory result, 2026-09-11.

## Executive conclusion

Pythia should be a post-training measurement system, not an AutoML system. Its
input is a frozen evaluation artifact: outcomes, aligned honest out-of-fold
predictions, entity/fold identifiers, representation provenance, and an
optional versioned guard. Its output is a tiered attainable-risk statement,
dependence sensitivity, diagnostics, or a refusal. Pythia never fits, tunes,
embeds, prompts, or engineers a predictor.

The most defensible theoretical direction is not to rename mutual information.
The basic loss-specific value of adding a representation is classical Bayesian
decision theory. The possible contribution is a *certification calculus* that
combines attainable-risk intervals, declared information/garbling relations,
learner-gap decomposition, and shared-error sensitivity. That contribution is
only a theorem candidate until the proof and literature audits are complete.

The first controlled simulation gives a useful but deliberately modest result.
Across 15 known-risk Gaussian representation cases, a naive orthogonality bound
was below the analytic Bayes risk in 6/15 cases. The dependence-aware curve,
evaluated at the simulation-known shared-error correlation cap, was valid in
15/15 cases. Its median bound/Bayes-risk ratio was only 0.197. Thus it rescued
validity under shared bias, but did not yet establish competitive tightness.

## 1. Object being estimated

For a loss `ell`, outcome `Y`, and information set `X`, define attainable Bayes
risk

```text
R*_ell(X) = inf_a E[ell(Y, a(X))].
```

For an added signal `Z`, define loss-specific attainable information

```text
I_ell(Z -> Y | X) = R*_ell(X) - R*_ell(X,Z).
```

This is nonnegative when `(X,Z)` genuinely contains `X` and the action class is
unrestricted. It has the operational units of the chosen loss: minutes under
MAE, squared minutes under MSE, and nats under log loss. That unit preservation
is valuable clinically, but the underlying value-of-information idea is not a
novel invention. Blackwell's comparison of experiments orders information
structures by decision value; a garbling cannot improve Bayes risk for a fixed
decision problem. Predictive V-information instead asks what information is
usable by a restricted predictive family and has been used to measure dataset
difficulty. Pythia must cite both rather than imply that any nonnegative
risk-difference definition is new.

The terminology in code is provisionally “decision information.” A publication
should only coin a stronger name if a literature review demonstrates that the
interval-valued construction below is not already known.

## 2. Information-order certificate calculus

Representations form a directed acyclic graph. An edge `rich -> poor` asserts a
real information relation: `poor` is a deterministic or randomized garbling of
`rich`. Examples include a note followed by a predeclared erasure/masking
channel, or `(tabular, notes)` projected to `tabular`. Arbitrary independently
trained feature sets do not automatically justify an edge.

Suppose every node `v` has a jointly valid interval

```text
l_v <= R*_ell(v) <= u_v.
```

For `rich -> poor`, Blackwell monotonicity gives

```text
R*_ell(rich) <= R*_ell(poor).
```

The order closure is therefore

```text
l'_v = max lower bound among ancestors of v
u'_v = min upper bound among descendants of v.
```

If all source intervals cover simultaneously and every edge is correct, the
closed intervals also cover simultaneously: each propagated inequality follows
from transitivity of the order. If closure produces `l'_v > u'_v`, Pythia must
refuse rather than average away the contradiction. Point estimates are projected
onto the same order cone with weighted isotonic regression, but this projection
does not upgrade their claim tier.

For a valid path `rich -> ... -> poor`, attainable information has interval

```text
[max(0, l_poor-u_rich), max(0, u_poor-l_rich)].
```

The current implementation is in `pythia/information_order.py`. Its theorem is a
straightforward consequence of the supplied simultaneous intervals and the
declared order; the hard scientific problem remains constructing informative
finite-sample intervals for `R*`.

## 3. Separating information from learner quality

For any evaluated predictor `f_X`, write

```text
L(f_X) = R*_ell(X) + H(f_X; X),   H >= 0,
```

where `H` is the learner gap. For nested information sets,

```text
L(f_poor) - L(f_rich)
  = [R*(poor) - R*(rich)] + [H_poor - H_rich].
```

This identity answers a recurring feature-engineering confusion. A richer
representation cannot increase unrestricted Bayes risk, but a particular
learner can perform worse because estimation, optimization, regularization, or
distribution shift increased its learner gap. Character-shuffled notes are a
good falsification control for the fitted NLP pipeline; they are not proof that
the original text had zero information. Conversely, inserting the outcome into
text creates label leakage and should collapse attainable risk toward zero; it
tests whether the system detects a suspiciously powerful information channel,
not whether the feature is deployable.

This decomposition is the best candidate for a useful new Pythia object: a
loss-specific *interval-valued attainable information statement* paired with an
interval for learner-gap movement. Novelty is not asserted yet.

## 4. Shared-error dependence

Disagreement estimators fail when candidate predictors share approximation
bias. For squared loss, Pythia currently constructs the error-covariance matrix
compatible with observed predictor risks and pairwise disagreements, then asks
for the smallest attainable risk compatible with positive semidefiniteness and
a declared cap `lambda_max` on positive shared-error correlation. The output is
a monotone sensitivity curve `B(lambda_max)`, a feasible dependence interval,
active constraints, and refusal diagnostics.

This target differs from two recent disagreement papers. *Bound to Disagree*
uses a certifiable surrogate and unlabeled disagreement to bound a target
predictor's ordinary true risk. *Model Agreement via Anchoring* studies expected
squared disagreement between independently trained models and proves controls
for particular learning algorithms, with extensions to strongly convex losses.
Pythia instead seeks a lower bound on the representation's attainable/Bayes
floor under an explicit shared-error sensitivity cap. The distinction is
plausible, but a full theorem and related-work audit are required before calling
it novel.

The dependence profile remains `exploratory` because empirical OOF moments
introduce sampling error and the population proof has not been externally
audited. No MSE argument will be transferred to MAE by analogy. MAE needs its
own guard and dependence calibration.

### Implemented sharpening after the first audit

The current development branch now contains two simpler, auditable squared-loss
rungs. The first solves the exact two-predictor population correlation inequality
and is sharp when the active pair reaches the declared correlation cap. The
second combines a conservative pairwise inequality with simultaneous one-sided
empirical-Bernstein confidence limits. Its proof and applicability boundary are
in `docs/theory/DEPENDENCE_BOUND.md`.

A frozen 1,800-run bounded-i.i.d. comparison (seed 20260912) found:

| method | observed coverage | 95% Wilson lower | median tightness |
|---|---:|---:|---:|
| legacy empirical orthogonality bound | 0.657 | 0.634 | invalid overall |
| Hoeffding finite-sample rung | 1.000 | 0.9979 | 0.000 overall; 0.256 at n=10,000 |
| empirical-Bernstein finite-sample rung | 1.000 | 0.9979 | 0.581 overall; 0.887 at n=10,000 |

The sharp plug-in estimator had 1.77% median relative error, compared with 9.54%
for the legacy disagreement estimator. This supports continued development, but
only for the tested bounded-i.i.d. designs with the true correlation and error
caps supplied. It is not yet a universal or OOF guarantee.

## 5. Relation to accepted approaches

The comparator harness must compare estimands before comparing numbers.

| Method | Primary estimand | Uses trained models? | Immediate role |
|---|---|---:|---|
| Pythia Bregman/max-pair | lower bound on attainable risk under orthogonality | frozen predictions only | baseline |
| Pythia dependence profile | attainable-risk sensitivity under shared-error cap | frozen predictions only | theorem candidate |
| KNIFE-P / LMC-P | conditional-entropy estimates transformed into regression predictability | estimator implementation trains | external comparator |
| Predictive V-information | information usable by a declared predictive family | usually yes | conceptual and empirical comparator |
| Conformal risk control | expected monotone loss of a selected prediction-set procedure | post-hoc calibration | coverage machinery, not direct Bayes-floor estimator |
| Bound to Disagree | ordinary risk of target via certifiable surrogate | surrogate required | disagreement comparator with different target |
| Anchoring | independent model disagreement | algorithm-specific analyses | dependence/independence comparator |

KNIFE-P and LMC-P specifically target regression predictability via conditional
entropy estimators that under- and over-estimate entropy. They are important
comparators, but they cannot be reimplemented from a verbal description: the
official method/code and hyperparameter protocol must be frozen first. Conformal
risk control can help turn a monotone calibration family into finite-sample risk
control, but its standard guarantee is not automatically a lower confidence
bound on Bayes risk. CAAFE and MLE-STAR concern automated feature/model search;
they belong to Horos/Metis and are upstream of Pythia.

## 6. Training-free system boundary

```text
Horos or external benchmark
  raw data -> representations -> train/tune within development folds
           -> frozen OOF prediction artifact + provenance
                                      |
                                      v
Pythia
  validate alignment -> route loss -> calculate bound/estimate/profile
                     -> close information order -> certify or refuse
```

Pythia may generate mathematical population fixtures for its own unit tests, but
it does not choose or fit predictive mappings. The architecture regression test
parses every Python file under `pythia/` and rejects `.fit`, `.partial_fit`,
`.fit_transform`, and imports from common trainer modules. Training-based M0 and
Mayo scripts now live under `horos/benchmark` and `horos/models`.

For real inputs, a future artifact contract must require:

- row IDs and outcomes aligned one-to-one;
- prediction columns declared as OOF, external holdout, or prospective;
- fold and entity identifiers;
- learner family and training-recipe hashes;
- representation source hash and timing/provenance audit;
- no hidden retraining inside `certify()`.

Pythia cannot prove from numbers alone that predictions are honest OOF values.
It can fail closed when the provenance declaration is absent or inconsistent.

## 7. Synthetic program before Mayo

### Stage T1: analytic tabular Gaussian chain

Generate independent standard normals `X1, X2` and

```text
Y = X1 + beta X2 + epsilon, epsilon ~ Normal(0, sigma^2).
```

The tabular set observes `X1`. The intact-note set observes `X1,X2`. A noisy note
passes `X2` through an independent erasure channel with probability `p`. Exact
MSE Bayes risks are

```text
R*(tabular) = sigma^2 + beta^2
R*(erased note) = sigma^2 + p beta^2
R*(intact note) = sigma^2.
```

This design separately varies target noise, information value, erasure, and
shared predictor bias. Analytic Bayes acts plus controlled errors create frozen
prediction matrices; no learner is fitted by Pythia.

First run (seed 20260911, 40,000 observations per design, five designs and three
representations):

| diagnostic | result |
|---|---:|
| representation cases | 15 |
| naive orthogonality bound valid vs population risk | 6/15 |
| dependence bound valid at true cap vs population risk | 15/15 |
| dependence bound valid vs empirical oracle risk | 15/15 |
| median dependence bound / population Bayes risk | 0.197 |

This is an exploratory regression check, not a finite-sample guarantee and not
evidence that Pythia is the tightest method.

### Stage T2: distributional stress

Add Laplace, Student-t, skewed mixture, heteroskedastic, sparse interaction, high
dimension, and group-correlated designs. Vary common model bias independently
of outcome noise and signal-to-noise ratio. Include wrong `lambda_max`, wrong
garbling edges, row misalignment, duplicated patients, and label leakage as
falsification tests. A theorem-conditioned result must have zero population
violations when assumptions are satisfied and must downgrade/refuse outside its
domain.

### Stage T3: generated language channel

Generate notes from latent variables using controlled templates with synonyms,
negation, distractors, contradictions, missingness, and temporal leakage. The
generator records the exact channel from latent state to text. Horos creates
OOF predictions using fixed classical and modern text pipelines; Pythia sees
only prediction matrices and hashes. Perturbations include erasure, segment
dropout, duplication, token shuffle, and character shuffle. Only perturbations
that are genuine garblings receive information-order edges.

### Stage T4: external comparator freeze

Implement or vendor exact, cited versions of KNIFE-P/LMC-P and the predeclared
classical variance/disagreement baselines. Run all methods on untouched seeds.
Report coverage, Wilson lower confidence bound, median tightness, interval width,
refusal rate, runtime, and paired differences. “Significant win” requires both:

1. validity gate: the one-sided 95% Wilson lower bound meets the predeclared
   target (and zero population violations for theorem-conditioned designs);
2. efficiency gate: materially tighter valid bounds or lower estimator error on
   held-out designs under paired inference and multiplicity control.

No method wins by returning an invalid aggressive number. Refusals remain in the
denominator and are reported separately.

### Stage T5: transfer to Mayo

Only after T1-T4 pass: freeze the method registry, then use Mayo as a
retrospective application. Horos owns feature construction, NLP, and model
training. Pythia consumes exact-cohort OOF artifacts. Mayo cannot be used to tune
constants that later support a Mayo claim.

## 8. Immediate proof obligations

1. Write a population theorem for the squared-loss covariance feasibility
   program, including necessity/sufficiency and the exact role of nonnegative
   versus absolute correlations.
2. Add sampling uncertainty to observed risks and disagreements, preferably as
   a joint confidence region rather than plug-in tolerances.
3. Prove monotonicity of `B(lambda)` analytically; retain numerical monotone
   closure only as a safety check.
4. Prove the information-order interval closure and state simultaneous-coverage
   requirements formally.
5. Audit isotonic projection literature before claiming novelty or risk
   improvement for projected point estimates.
6. Develop MAE dependence separately and keep it calibrated until an independent
   proof exists.

## Sources

1. David Blackwell, “Equivalent Comparisons of Experiments” and related 1953
   decision-theoretic work; bibliographic archive:
   https://digicoll.lib.berkeley.edu/record/112749
2. Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta, “Understanding Dataset
   Difficulty with V-Usable Information,” ICML 2022:
   https://proceedings.mlr.press/v162/ethayarajh22a.html
3. Yu-Hsueh Fang and Chia-Yen Lee, “Predictability Analysis of Regression
   Problems via Conditional Entropy Estimations,” 2024:
   https://arxiv.org/abs/2406.03824
4. Mathieu Bazinet, Valentina Zantedeschi, and Pascal Germain, “Bound to
   Disagree,” UAI 2026: https://proceedings.mlr.press/v337/bazinet26a.html
5. Eric Eaton et al., “Model Agreement via Anchoring,” COLT 2026:
   https://proceedings.mlr.press/v336/eaton26a.html
6. Anastasios N. Angelopoulos et al., “Conformal Risk Control,” ICLR 2024:
   https://arxiv.org/abs/2208.02814
7. Noah Hollmann, Samuel Müller, and Frank Hutter, “CAAFE,” NeurIPS 2023:
   https://papers.nips.cc/paper/2023/hash/8c2df4c35cdbee764ebb9e9d0acd5197-Abstract-Conference.html
8. Jaehyun Nam et al., “MLE-STAR,” 2025: https://arxiv.org/abs/2506.15692
