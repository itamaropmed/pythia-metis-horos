# Finite decision information: a statistical representation-gap edge

## Result

Pythia now has a model-free, statistically justified upper bound on the value
of a **finite registered representation refinement**.  It answers the question
needed by the certificate graph:

```
R*(G) - R*(G,Z) <= ?
```

without declaring `Z` irrelevant and without training a predictor.

The useful route is loss-relative.  A generic conditional-mutual-information
upper bound was implemented first and found valid but loose.  A direct
decision-information endpoint then beat the CMI/Pinsker route in every one of
150 untouched comparisons and in every one of 300 fresh multi-loss cells,
with no known-gap exceedances.

This is a finite-alphabet theorem.  It does not certify arbitrary continuous
features or raw strings, and it refuses outcome binning unless the binned
outcome is itself the registered endpoint.

## The direct theorem

Let `P` be the joint law of finite `(Y,Z,G)`, let `P_hat` be its empirical law,
and suppose one joint-law confidence event gives

```
TV(P,P_hat) <= delta.
```

For a finite action set and loss matrix with range `A`, define the Bayes risks
under the coarse representation `G` and refinement `(G,Z)` as `R_G(P)` and
`R_GZ(P)`.  For either representation,

```
|R(P)-R(Q)| <= A TV(P,Q).
```

This follows because the expectation of every fixed decision rule is
`A`-Lipschitz in total variation, and taking an infimum over the same action
set preserves that modulus.  Therefore, on the same confidence event,

```
0 <= Delta_loss(P)
   = R_G(P) - R_GZ(P)
  <= Delta_loss(P_hat) + 2 A delta.
```

`finite_decision_gap_upper_bound` uses the fixed-sample multinomial L1 bound

```
P(||P_hat-P||_1 >= epsilon)
    <= 2^K exp(-n epsilon^2/2),
```

where `K=|Y||Z||G|`, and sets `delta=epsilon/2`.  It evaluates the empirical
Bayes risks exactly over every declared action column.  The resulting gap cap
enters `pythia.certificate_graph` as a statistical edge and spends its recorded
alpha once.

The proof is a direct composition of classical multinomial concentration and
Bayes-risk stability, not a claim that total variation or value of information
is newly invented.  The potentially publishable systems contribution is the
loss-relative, executable edge inside a fail-closed certificate compiler.

## Why CMI testing is not enough

Recent finite-discrete conditional-information work studies testing
`I(Y;Z|G)=0` versus separation by at least `epsilon`; see
[Seyfried, Sen & Tomamichel (COLT 2025)](https://proceedings.mlr.press/v291/seyfried25a.html)
and the matching lower-bound programme in
[Seyfried et al. (COLT 2026)](https://proceedings.mlr.press/v336/seyfried26a.html).
Failure to reject the null is not an upper confidence bound on the decision
value and therefore cannot justify a zero edge in Pythia.

For comparison, `finite_cmi_upper_bound` combines the same joint-law L1 event,
TV contraction, the
[Fannes--Audenaert entropy modulus](https://arxiv.org/abs/quant-ph/0610146),
and Pinsker:

```
Delta_loss <= A sqrt(I(Y;Z|G)/2).
```

It is valid and remains available as a general information diagnostic.  The
direct theorem avoids controlling four entropy terms and then translating a
distributional quantity back to one loss.

## First ten-law audit: sample-size behavior

The first audit used seeds `128001--128005` and was development evidence for
the CMI route.  Ten fixed three-outcome/two-refinement/two-group laws ranged
from conditional independence through rare signal to near target disclosure.

| n | CMI coverage misses | Nontrivial CMI UCB | Median CMI UCB | Median Pinsker gap cap | Median true 0-1 gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 0/50 | 0/50 | 0.693 | 0.589 | 0.045 |
| 30,000 | 0/50 | 40/50 | 0.449 | 0.474 | 0.045 |
| 300,000 | 0/50 | 45/50 | 0.221 | 0.332 | 0.045 |

At a Mayo-like `n=3,000`, the generic route was exactly the trivial alphabet
cap in every run.  This motivated the direct loss-relative theorem; the same
seeds were not used as confirmation for it.

## Untouched direct-gap confirmation

Seeds `129001--129005` were untouched after the direct theorem and candidate
rule were fixed.

| n | Direct strict wins over CMI/Pinsker | Mechanism wins | Direct misses | Pinsker misses | Median direct cap | Median Pinsker cap | Median truth |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | **50/50** | **10/10** | 0 | 0 | **0.133** | 0.589 | 0.045 |
| 30,000 | **50/50** | **10/10** | 0 | 0 | **0.073** | 0.473 | 0.045 |
| 300,000 | **50/50** | **10/10** | 0 | 0 | **0.054** | 0.332 | 0.045 |

At `n=3,000`, conditionally independent laws received a median direct cap
`0.0868`, while near-disclosure received `0.5618`; thus the cap reacts in the
required direction to added target information.  The corresponding Pinsker
cap was `0.5887` for both because its CMI endpoint saturated.

Ten wins out of ten registered mechanisms has the descriptive one-sided sign
reference `2^-10 = 0.0009766`.  The mechanisms are designed stress cases, not
an iid sample from all possible tasks, so this is not a universal-superiority
probability.

Raw rows:
`finite_decision_gap_confirmation_results.jsonl`, SHA-256
`21cd71437868219f26fe99ff55f746c008d25bd076bcf6afaa9a9428bc045b16`.

## Untouched multi-loss confirmation

Seeds `130001--130005`, `n=3,000`, six losses, and all 10 mechanisms were
frozen with family alpha `0.05/6` per loss.  The action set was the full
three-point outcome alphabet.

| Loss | Direct strict wins | Mechanism wins | Direct misses | Pinsker misses | Median direct cap | Median Pinsker cap | Median true gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE | **50/50** | **10/10** | 0 | 0 | **0.286** | 1.177 | 0.109 |
| MSE | **50/50** | **10/10** | 0 | 0 | **0.511** | 2.355 | 0.135 |
| Q10 | **50/50** | **10/10** | 0 | 0 | **0.172** | 1.060 | 0.005 |
| Q50 | **50/50** | **10/10** | 0 | 0 | **0.143** | 0.589 | 0.055 |
| Q90 | **50/50** | **10/10** | 0 | 0 | **0.176** | 1.060 | 0.008 |
| Exact-window miss | **50/50** | **10/10** | 0 | 0 | **0.139** | 0.589 | 0.045 |

Raw rows:
`finite_decision_multiloss_confirmation_results.jsonl`, SHA-256
`f6bd1d058ba699273142a82a99df4579f9ed539cf94049ae7da635fae550ec7f`.

The strong result is comparative tightness against the matched generic
information route, not proof that the direct cap is minimax optimal.  Q10/Q90
remain conservative when their true decision value is close to zero.

## Mayo and text boundary

This edge can apply to a predeclared finite concept ladder when:

- `G`, `Z`, the outcome alphabet, and the complete action/loss matrix are
  frozen before evaluation;
- all categories, including unobserved ones, are listed;
- observations are iid or an appropriately replaced cluster-level joint-law
  concentration theorem is supplied;
- the finite outcome is the actual clinical endpoint, not a convenient binning
  of a richer duration outcome.

It does not apply directly to arbitrary raw notes.  A token/string alphabet is
astronomical, and silently mapping it into concepts changes the estimand.  For
Mayo duration on a large minute/action alphabet crossed with procedure and note
concept states, the present Weissman radius will usually saturate.  The next
research step is a cluster-aware, loss-specific confidence set that exploits
the loss matrix or product structure without paying for the full joint alphabet.

## Implementation

- `pythia/core/finite_cmi.py`: finite CMI and direct decision-gap endpoints;
- `pythia/evidence_adapters.py`: statistical gap-edge adapters;
- `run_finite_cmi_gap_ood.py`: ten-law sample-size audit;
- `run_finite_decision_multiloss_ood.py`: untouched multi-loss audit;
- `tests/test_finite_cmi.py`: coverage, leakage, refusal, multi-loss, and graph
  integration tests.

All audit data are synthetic.  No patient-derived data or text are stored.
