# Original-scale SAA certificates: moment routes and safe score truncation

## Decision

Pythia can lower-bound an **original-outcome, unbounded-loss** Bayes risk without
clipping the outcomes.  The strongest tested route is not a new heavy-tail mean
inequality.  It is the simpler composition

1. compute a nonnegative empirical-optimum (SAA) score on original outcomes;
2. truncate the **score**, not the outcome, at a predeclared cap;
3. lower-bound the truncated score mean with the published dKelly e-value;
4. select over a finite route/cap portfolio under one explicit alpha ledger.

Because `min(S,c) <= S` and SAA optimism gives `E[S] <= R*`, every bounded-score
endpoint remains a lower bound on the original Bayes risk.  No outcome tail,
moment, or common-mean assumption is required.  This is now implemented in
`pythia.core.truncated_saa` and exported by the public package.

The separately implemented Cantelli and Catoni-style declared-moment routes are
valid, but they were weaker in the frozen confirmation below.  They remain
diagnostics/fallbacks, not the production default.

## Theorems and assumption boundary

Let `S_i >= 0` be mutually independent SAA scores.  They may be non-identically
distributed.  Let `r_i` be their corresponding Bayes risks, so that
`E[S_i] <= r_i`.  For a fixed cap `c`, define `X_i=min(S_i,c)`.  Then

```
average E[X_i] <= average E[S_i] <= average r_i.
```

Inverting the dKelly e-value for the bounded `X_i` therefore gives a lower
endpoint for the original average Bayes risk.  If routes or caps are selected
after observing their endpoints, Pythia assigns `alpha/(routes*caps)` to every
candidate and takes their maximum only on the simultaneous event.  Dependence
between candidates is allowed; independence is required within each candidate
score vector.

For comparison, `pythia.core.moment_saa` adds two conditional routes:

- one-sided Cantelli from declared score-variance caps;
- a Catoni polynomial-soft-truncation e-value from declared score second-moment
  caps.  Its proof directly targets `average_i E[S_i]`; it does not smuggle in
  an iid or common-mean assumption.

The location-loss helper derives conservative second-moment caps from moments
about a fixed, evaluation-independent action.  MAE and pinball need a finite
second outcome moment; MSE needs a finite fourth moment.  Thus Student-t3 MSE
correctly refuses the moment route.

## Frozen ten-mechanism confirmation

Script: `run_original_scale_moment_ood.py`  
Rows: `original_scale_catoni_confirmation_results.jsonl`  
SHA-256: `fd680e039a4a44cd73c537eedf811b02c735a0ac8738f0423a702142932d04e6`

- untouched seeds `127001--127005`;
- 10 registered noise laws, 5 runs per mechanism, `n=16,000`;
- losses MAE, Q90, and MSE share family alpha `0.05`;
- block sizes `k=(2,4,8,16)`;
- dKelly caps are predeclared multiples `(0.5,1,2,4,8)` of the population scale;
- gold risks and moment caps come from population integrals, never audit rows;
- no predictor or representation is trained.

| Loss | Catoni supported | Catoni median bound/gold | dKelly median bound/gold | dKelly strict wins | Gold exceedances (both) | Plug-in median relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE | 50/50 | 0.846 | **0.924** | 50/50 | 0 | 0.0061 |
| Q90 | 50/50 | 0.516 | **0.845** | 50/50 | 0 | 0.0065 |
| MSE | 45/50 | 0.805 | **0.862** | 50/50 | 0 | 0.0130 |

Cantelli was the registered negative control: median bound/gold was `0.566`
for MAE, `0.000` for Q90, and `0.385` for supported MSE cells.  Catoni improved
all three substantially, but did not beat dKelly in any confirmation cell.

### Per-mechanism median bound/gold (Catoni / dKelly)

| Mechanism | MAE | Q90 | MSE |
| --- | ---: | ---: | ---: |
| Normal | 0.848 / **0.933** | 0.494 / **0.872** | 0.831 / **0.899** |
| Laplace | 0.862 / **0.940** | 0.516 / **0.849** | 0.825 / **0.881** |
| Logistic | 0.852 / **0.935** | 0.503 / **0.867** | 0.830 / **0.892** |
| Student-t3 | 0.845 / **0.921** | 0.517 / **0.834** | refused / **0.757** |
| Student-t5 | 0.839 / **0.921** | 0.493 / **0.846** | 0.781 / **0.848** |
| Uniform | 0.839 / **0.928** | 0.489 / **0.918** | 0.837 / **0.918** |
| Gamma-2 | 0.839 / **0.919** | 0.545 / **0.835** | 0.795 / **0.861** |
| Exponential | 0.848 / **0.923** | 0.583 / **0.834** | 0.805 / **0.867** |
| Lognormal-0.8 | 0.851 / **0.921** | 0.599 / **0.804** | 0.744 / **0.796** |
| Contaminated normal | 0.846 / **0.912** | 0.552 / **0.826** | 0.742 / **0.767** |

## What this does and does not solve

This closes the avoidable outcome-clipping problem for SAA concentration: an
unbounded original-loss target can use bounded score evidence.  It also makes
the package more loss-generic because the concentration layer needs only a
nonnegative SAA score.

It does **not** manufacture replicates or valid near-replicate groups.  In an
ordinary one-outcome-per-case Mayo dataset, Pythia still needs exact groups, a
declared/externally bounded residual-law discrepancy, or another justified
representation-gap edge.  The declared-modulus attachment supplies a correct
conditional curve but not an identified Mayo discrepancy.  Its independently
trained HGB shift remains an external frozen alignment resource rather than
training inside Pythia.

The experiment is a noise-law stress test, not evidence that Pythia beats every
published Bayes-floor method.  The result instead chooses the best internally
audited concentration route and records the losing alternatives.

## Attachment provenance

The supplied note and archive were treated as evidence, not instructions:

- note SHA-256: `cf893fb48e1884c6d1b663a2f56cf9fccb22f56abeac37f05144ae39aa7bb50f`;
- archive SHA-256: `43166b4056868fd0b36e58ade39af84e67da76b31ff78113ac72844053a4eb95`.

The bundle's barycentric theorem agrees with the already audited implementation.
Its aggregate discrepancy remains declared/oracle in the synthetic theorem
check, and the 31.2623 Mayo incumbent remains unreproduced in the strict Mayo
manifest.  No Mayo headline number is changed by this work.
