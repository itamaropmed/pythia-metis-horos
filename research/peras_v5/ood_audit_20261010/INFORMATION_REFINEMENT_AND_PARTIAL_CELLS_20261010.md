# Information-budget refinement and partial-cell certificates

## Research question

Can a model-training-free lower certificate for a coarse representation `G` remain valid for a richer registered representation `(G,Z)` without simply assuming that `Z` is useless?

Two complementary routes were implemented and tested:

1. an information-budget transfer, valid under an upper bound on `I(Y;Z|G)`;
2. a partial fine-cell certificate that directly estimates sufficiently replicated `(G,Z)` cells and assigns zero to unresolved target mass.

The first is broadly valid but was vacuous on the fresh OOD battery. The second remained useful until fine-cell replication collapsed. This negative/positive pair is the main result: generic information inequalities are a safety fallback, while direct replicated geometry is the practical route.

## Theorem 1: conditional-information transfer

Let `P_g` denote the clipped conditional outcome law given `G=g`, and `P_gz` the law after observing a finite pre-outcome refinement `Z=z`. Let

```
R_G = E_g inf_a E_{P_g} loss(Y,a)
R_GZ = E_gz inf_a E_{P_gz} loss(Y,a).
```

If the clipped loss has range at most `A`, then

```
0 <= R_G - R_GZ
   <= A E_gz TV(P_gz,P_g)
   <= A sqrt(I(Y;Z|G)/2).
```

The first inequality is Blackwell monotonicity. For the second, insert the action optimal for `P_gz` into the coarse problem and use the bounded-expectation/total-variation inequality. Pinsker followed by Jensen yields the last line. For finite `Z`, `I(Y;Z|G) <= H(Z|G)`. Therefore a coarse lower confidence bound `B_G` transfers as

```
B_GZ(J) = max(0, B_G - min(M_loss, A sqrt(J/2))),
```

whenever `J` is a valid upper bound on conditional mutual information and `M_loss` is the distribution-free maximum clipped Bayes risk.

For outcome-Lipschitz losses, if every coarse conditional law satisfies a declared `T1(c)` transport-information inequality, the additional cap

```
R_G - R_GZ <= L_loss sqrt(2 c J)
```

can be used. The implementation takes the minimum of all valid caps. It refuses this route for discontinuous window loss and refuses every outcome-derived or provenance-free refinement. It does not estimate mutual information or validate `c`.

This theorem is an application of classical comparison-of-experiments, Pinsker, and transport-information machinery—not a new information inequality. Reid and Williamson connect risk and divergences for binary experiments; Williamson and Cranko's 2024 information-risk bridge covers a substantially broader modern framework. Blackwell/Le Cam comparison and deficiency theory are the correct conceptual ancestors. The Pythia contribution, if retained, is executable assumption accounting and integration with finite-sample lower endpoints.

## Property verification

`check_information_refinement.py` generated 1,000 random finite joint laws over two coarse groups, four refinement states, and five outcomes. Across eight losses there were:

- 8,000 theorem cases;
- zero richer-risk bound exceedances;
- zero information-gap-cap exceedances;
- exact conditional-entropy identity checks;
- required refusal for outcome-derived refinements;
- required T1 refusal for window loss.

The proof supplies validity; these tests verify the implementation over heterogeneous random laws.

## Fresh OOD utility audit: valid but vacuous

`run_information_refinement_ood.py` froze ten exact-law duration mechanisms, five new seeds 98001--98005, 20,000 calibration outcomes, 800 target cases, and eight losses. `G=procedure code`; `Z=pre-outcome note concept`; the finer `(code,concept)` key is an exact conditional-law key by construction. Median registered `H(Z|G)` was 0.998 nats, ranging from 0.904 to 1.047.

The entropy/Pinsker transferred lower bound was positive in **0/400** loss-dataset cases. It never exceeded known fine-representation gold, but it was useless. Direct grouped SAA for the fine representation remained positive in all 50 runs for MAE, MSE, Q10, Q50, Q90, and W15; 43/50 W30 and 2/50 W60 runs were positive.

| Loss | Information transfer positive / 50 | Direct fine positive / 50 | Direct fine median lower/gold | Gold exceedances, either route |
| --- | ---: | ---: | ---: | ---: |
| MAE | 0 | 50 | 0.857 | 0 |
| MSE | 0 | 50 | 0.706 | 0 |
| Q10 | 0 | 50 | 0.787 | 0 |
| Q50 | 0 | 50 | 0.857 | 0 |
| Q90 | 0 | 50 | 0.763 | 0 |
| W15 | 0 | 50 | 0.655 | 0 |
| W30 | 0 | 43 | 0.405 | 0 |
| W60 | 0 | 2 | 0.000 | 0 |

This is a useful rejection of an attractive but weak idea: conditional entropy alone does not make a tight distribution-free Pythia certificate at Mayo-like outcome scales.

## Theorem 2: partial fine-cell lower certificate

For each target fine group `h=(g,z)` with at least `k` independent calibration outcomes, partition calibration outcomes into disjoint blocks and compute the in-block empirical optimal loss `V_hb`. Sample-average-approximation optimism gives

```
E[V_hb] <= R_h.
```

Clipping this nonnegative witness from above at a registered score cap preserves the inequality. A weighted Hoeffding bound over all included independent blocks gives a lower confidence bound on the included groups' weighted risk. Missing or undersized fine groups contribute zero; since risks are nonnegative, the result remains a lower bound on the **full** fine-representation Bayes risk. Cell inclusion is allowed to use calibration counts, but never outcomes.

`partial_group_lower` reports included and omitted target mass, group counts, block counts, radius, assumptions, and the declared fine-representation target. It does not reinterpret a partial-cell result as a raw-text certificate.

In 500 discrete known-floor repetitions across eight losses—4,000 cases total—with 40% target mass deliberately missing or undersized, there were zero gold exceedances. There were 1,557 positive bounds; conservative zeros are expected for low-risk/window cases.

## High-cardinality augmentation audit

`run_sparse_refinement_ood.py` adds an independent, pre-outcome nuisance bucket to the exact `(code,concept)` representation. Bucket counts 1, 2, 5, 10, and 20 were fixed and all 40 loss/augmentation endpoints share one Bonferroni 5% budget per generated dataset. Ten mechanisms and five new seeds 99001--99005 were used. The true fine-representation Bayes risk is unchanged because the nuisance is independent by construction, but the partial route does not use that oracle independence—it relies only on exact fine-cell calibration.

All 2,000 lower endpoints stayed below known gold.

| Buckets | Median covered target mass | MAE positive / 50 | MAE median lower/gold | Q90 positive / 50 | Q90 median lower/gold | W30 positive / 50 | W30 median lower/gold |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 50 | 0.848 | 50 | 0.764 | 41 | 0.385 |
| 2 | 1.000 | 50 | 0.836 | 50 | 0.752 | 40 | 0.350 |
| 5 | 0.971 | 50 | 0.786 | 50 | 0.710 | 40 | 0.228 |
| 10 | 0.676 | 50 | 0.526 | 50 | 0.474 | 30 | 0.092 |
| 20 | 0.052 | 35 | 0.026 | 35 | 0.025 | 10 | 0.000 |

MSE, Q10, Q50, and W15 show the same monotone loss of useful tightness as replicated target mass disappears. W60 remains essentially zero at every resolution because its attainable miss risk is already near zero.

This is the desired fail-closed response to arbitrary engineered refinements: Pythia does not declare the nuisance harmless from its name, and it does not extrapolate across unseen cells. If the augmentation mechanism itself provides a defensible structural statement `Y independent of Z given G`, then the information gap is exactly zero and the coarse certificate transfers unchanged. That structural fact must come from provenance/design, not from a failed predictive model or a non-significant test.

## Selection-safe sparse-cell portfolio

A fixed `k=16` needlessly discards cells with 2--15 calibration outcomes. `partial_group_portfolio` predeclares block sizes `(2,4,8,16)`, gives each route one quarter of the loss/bucket alpha allocation, and selects the largest simultaneously covered lower endpoint. Selection may inspect the resulting scores because Bonferroni coverage holds for every candidate route at once.

In 500 discrete repetitions and eight losses, the portfolio had zero known-gold exceedances over 4,000 cases. It produced 2,379 positive bounds versus 1,557 for the fixed `k=16` route and was strictly tighter in 2,282 cases.

A further untouched battery used ten mechanisms, seeds 100001--100005, nuisance resolutions 5/10/20, and eight losses. Each portfolio receives `0.05/(8*3)` family alpha and divides it over four block sizes. The fixed comparator receives the same loss/bucket family alpha without selection. These are two separately valid methods; their uncorrected maximum is not reported.

| Buckets | Loss | Portfolio wins / 50 | Fixed wins / 50 | Portfolio positive | Fixed positive | Portfolio median lower/gold | Fixed median lower/gold | Portfolio median mass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | MAE | 31 | 19 | 50 | 50 | 0.802 | 0.792 | 1.000 |
| 5 | Q90 | 15 | 35 | 50 | 50 | 0.710 | 0.719 | 0.974 |
| 5 | W30 | 14 | 26 | 40 | 40 | 0.260 | 0.232 | 1.000 |
| 10 | MAE | 45 | 5 | 50 | 50 | 0.753 | 0.533 | 0.963 |
| 10 | Q90 | 35 | 15 | 50 | 50 | 0.600 | 0.480 | 0.954 |
| 10 | W30 | 34 | 6 | 40 | 30 | 0.168 | 0.096 | 0.968 |
| 20 | MAE | **50** | 0 | 50 | 35 | **0.660** | 0.025 | 0.937 |
| 20 | Q90 | **45** | 5 | 50 | 35 | **0.407** | 0.025 | 0.933 |
| 20 | W30 | 34 | 1 | 34 | 12 | 0.052 | 0.000 | 0.935 |

Across all 1,200 new loss endpoints the portfolio had zero known-gold exceedances. MSE, Q10, Q50, and W15 also improved sharply at 10 and 20 buckets; W60 remained zero. At five buckets the selection penalty can make the fixed route stronger, especially for Q90 and W15. The router should therefore choose the portfolio prospectively based on expected sparsity, or allocate alpha to both if both are candidates; it must not choose the favorable table after observing outcomes.

For a positive-mass fine cell with **no** outcome observations and no restriction on its conditional law, zero is minimax-unavoidable: two data-generating laws can agree on all observed data and covariates while making that cell deterministic under one law and noisy under the other. No universally valid procedure can assert a positive contribution for the unseen cell. The portfolio improves merely sparse cells; it does not evade this identification boundary.

## Updated router

For a registered representation refinement, the principled ordering is now:

1. **Structural zero gap:** preserve the coarse certificate only when the data-generating/provenance contract establishes conditional irrelevance.
2. **Direct full fine-group SAA:** use when every positive-weight fine group is replicated.
3. **Partial fine-cell SAA:** retain verified cell contributions and expose omitted target mass.
4. **Information sensitivity:** report `B(J)` when an external conditional-information or T1 cap is defensible.
5. **Localized exceptional-mass sensitivity:** report `B(epsilon)` when sufficiency failures are confined to an externally bounded target subset.
6. **Refuse:** raw text, unique identifiers, post-outcome fields, unsupported losses, or unvalidated caps receive no full-information numeric certificate.

Selection among multiple statistical lower endpoints requires a shared alpha allocation. Structural and deterministic sensitivity caps do not consume statistical alpha, but their provenance remains a substantive assumption.

## Novelty and next gap

Neither Pinsker transfer nor SAA optimism is new. The strongest paper direction is a **certificate compiler for nested information structures**: it routes among exact replication, partial replication, transport-corrected near replication, and explicit sensitivity while maintaining one assumption/error ledger. The scientific gap is a useful, auditable upper bound on the value of continuous or raw-text refinement. Entropy-only control is too loose, and distribution-free finite data cannot establish raw-feature sufficiency.

The next justified development target is a loss-specific conditional-law discrepancy certificate with sample splitting: estimate an upper cap on average fine-to-coarse transport using independent repeated environments or known randomization, then combine it with the direct partial-cell contribution. Without repeated environments, a structural generative assumption, or a restricted smoothness class, Pythia should refuse rather than claim universality.

## Reproducibility

- Theory/API: `refinement_gap_profile.py`, `finite_group_multiloss_interval.py`
- Property checks: `check_information_refinement.py`, `check_partial_group_lower.py`
- Fresh runners: `run_information_refinement_ood.py`, `run_sparse_refinement_ood.py`, `run_sparse_k_portfolio_ood.py`
- Raw results: `information_refinement_ood_results.jsonl`, `sparse_refinement_ood_results.jsonl`, `sparse_k_portfolio_ood_results.jsonl`

All data in these audits are generated. No Mayo or patient-derived content is stored.
