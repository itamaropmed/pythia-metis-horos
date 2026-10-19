# Outcome-alphabet-free finite-policy information gaps

## Result

The finite decision-information edge no longer needs a finite outcome
alphabet.  It now supports arbitrary outcomes whenever:

- coarse and refinement states are finite and fully registered;
- the action class is finite and contains a Bayes action for every supported
  conditional law;
- the loss is almost surely bounded on that action/outcome domain;
- rows are iid and the registry was frozen before evaluation.

On untouched seeds, the route was strictly tighter than both the earlier joint
multinomial decision bound and CMI/Pinsker in all 300 multi-loss cells, with
zero known-gap exceedances.

## Theorem

Let `F_G` contain every policy mapping a coarse state `G` to one of `A`
registered actions, and let `F_GZ` contain every policy mapping `(G,Z)` to an
action.  For loss in an interval of width `M`, define

```
R_G*  = inf_{f in F_G}  E loss(Y,f(G)),
R_GZ* = inf_{h in F_GZ} E loss(Y,h(G,Z)).
```

One-sided Hoeffding plus a union bound gives, simultaneously,

```
R_G*  <= Rhat_G*  + epsilon_G,
R_GZ* >= Rhat_GZ* - epsilon_GZ,
```

where

```
epsilon_G  = M sqrt((|G| log |A| + log(1/alpha_G))/(2n)),
epsilon_GZ = M sqrt((|G||Z| log |A| + log(1/alpha_GZ))/(2n)).
```

Therefore

```
R_G* - R_GZ*
    <= (Rhat_G* - Rhat_GZ*) + epsilon_G + epsilon_GZ.
```

Pythia chooses `alpha_G + alpha_GZ = alpha` by a deterministic scalar
optimization using only `n`, alphabet sizes, action count, loss range, and
alpha.  No outcomes enter the allocation.

The action-completeness requirement is substantive.  A restricted action
class can change the information gap in either direction, so the API refuses
to transfer a certificate unless completeness is declared and justified.  For
integer-valued MAE/pinball outcomes, an action grid containing the complete
outcome support is sufficient; MSE generally needs a different continuous-
action theorem or an explicit approximation correction.

This is classical finite-class uniform convergence composed around a
loss-relative value-of-information estimand.  The inequality itself should not
be marketed as a new concentration theorem.  Its role as an automatically
compiled representation-gap edge is the stronger systems contribution.

## Untouched multi-loss audit

- seeds `131001--131005`;
- 10 finite refinement mechanisms;
- `n=3,000` per run;
- family alpha `.05/6` over six losses;
- three complete actions;
- no predictor training.

| Loss | Policy wins vs joint L1 | Mechanism wins | Policy misses | Median policy cap | Median joint-L1 cap | Median CMI/Pinsker | Median truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE | **50/50** | **10/10** | 0 | **0.258** | 0.293 | 1.177 | 0.109 |
| MSE | **50/50** | **10/10** | 0 | **0.436** | 0.504 | 2.355 | 0.135 |
| Q10 | **50/50** | **10/10** | 0 | **0.143** | 0.174 | 1.060 | 0.005 |
| Q50 | **50/50** | **10/10** | 0 | **0.129** | 0.146 | 0.589 | 0.055 |
| Q90 | **50/50** | **10/10** | 0 | **0.143** | 0.174 | 1.060 | 0.008 |
| Exact-window miss | **50/50** | **10/10** | 0 | **0.126** | 0.143 | 0.589 | 0.045 |

It also beat CMI/Pinsker in 50/50 cells for every loss.  The descriptive
mechanism-level sign reference for 10/10 is `2^-10=0.0009766`; the mechanisms
are registered stress cases, not a random population of tasks.

Raw rows: `finite_policy_multiloss_confirmation_results.jsonl`, SHA-256
`19ca63bee7c282a42b3a91187eb2ddb07b8b20dcc7a595370de1e1777a2cf40c`.

## Remaining tightness gap

The route removes dependence on `|Y|` but still pays `log |A|` for every
coarse/refined cell and uses worst-case loss range.  For Mayo MAE on a complete
0--900 minute action grid, this remains too expensive.  The next justified
improvement is variance-adaptive simultaneous risk rectangles or a
loss-geometry theorem for ordered location actions.  Any such route must be
tested on untouched mechanisms and cannot infer the action grid from
evaluation outcomes.

## Variance-adaptive loss router

A development comparison on seeds `131001--131005` found that simultaneous
cell/action empirical-Bernstein rectangles were weaker than finite-policy
Hoeffding for MAE, MSE, Q50, and exact-window miss, but materially stronger for
Q10 and Q90.  The following V1 router was frozen before new outcomes:

| Loss | Frozen route |
| --- | --- |
| MAE, MSE, Q50, exact-window miss | finite-policy Hoeffding |
| Q10, Q90 | coordinate empirical Bernstein |

On untouched seeds `132001--132005`, family alpha `.05/6`, the router had zero
known-gap misses across all 300 cells.  It beat the joint-L1 cap in 50/50 runs
for every loss.  For its two non-default branches:

| Loss | Router wins over policy | Median routed cap | Median policy cap | Median joint-L1 cap | Median truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q10 | **50/50** | **0.093** | 0.143 | 0.174 | 0.005 |
| Q90 | **50/50** | **0.109** | 0.145 | 0.175 | 0.008 |

The other four routes equal their frozen policy endpoints and retain the
previous 50/50 wins over joint L1.  Raw rows:
`finite_gap_router_confirmation_results.jsonl`, SHA-256
`aee08939b6181676522546b6a748315317696ec7632544c3c3ee73b301656ad3`.

`finite_loss_routed_gap_upper_bound` exposes this as version
`finite-gap-v1`.  Unknown losses, including untested Q levels, refuse rather
than extrapolate the Q10/Q90 observation.  The route is selected from the loss
registry, never from current endpoints.

## Implementation

- `pythia/core/finite_policy_gap.py`;
- `finite_policy_gap_resolution`: outcome-blind Metis preflight;
- `finite_policy_gap_upper_bound`: statistical edge producer;
- `finite_coordinate_bernstein_gap_upper_bound`: variance-adaptive rectangle producer;
- `finite_loss_routed_gap_upper_bound`: frozen confirmed V1 loss router;
- `tests/test_finite_policy_gap.py`;
- `run_finite_decision_multiloss_ood.py`.
