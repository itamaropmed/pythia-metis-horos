# Repeated-environment coupling certificates for near-replicates

## Result

Pythia now has a non-oracle way to upper-bound the average conditional-law Wasserstein penalty **when independently repeated joint environments are available**. The method uses only replicated outcomes and a registered coupling design; it trains no predictor and does not inspect the evaluation outcomes used by the risk witness.

On ten near-replicate OOD mechanisms, five new seeds, and 5,000 calibration environments per dataset, a common-quantile coupling yielded positive MAE certificates in 46/50 runs, covering all ten mechanisms, with zero known-gold exceedances. Independent product coupling yielded 0/50 positive MAE bounds. Neither route produced a positive Q90 or MSE certificate. This is a proof of concept for an additional data-design resource, not a Mayo-ready solution.

## Coupling-cap theorem

Fix outcome-blind, globally disjoint target blocks. Unit `i` has a clipped conditional outcome law `Q_i`. Suppose environment `e` supplies a joint outcome vector

```
(Y_e1,...,Y_en) ~ Pi,
```

where coordinate `i` has marginal `Q_i`. Dependence within a row is arbitrary, but environment rows are independent and identically distributed. For each row define the average ordered within-block cost

```
C_e = (m k^2)^-1 sum_t sum_i sum_j |Y_eti - Y_etj|.
```

Because Wasserstein distance is the infimum of expected transport cost over all couplings,

```
eta = (m k^2)^-1 sum_tij W1(Q_ti,Q_tj) <= E[C_e].
```

After clipping to an interval of width `D`, `0 <= C_e <= D(k-1)/k`. A one-sided confidence upper bound for `E[C_e]` is therefore also an upper-confidence bound for `eta`. `calibrate_coupled_mean_w1` simultaneously computes Hoeffding, empirical-Bernstein, and transformed-Bentkus upper endpoints and takes their minimum after splitting the declared cap error budget across the three.

The resulting cap plugs directly into the existing disjoint tuple theorem. A separate error allocation covers every tuple-size/loss/score-cap risk endpoint. The total failure probability is at most the cap-family allocation plus the risk-witness allocation by a union bound.

This theorem is elementary optimal transport plus bounded-mean concentration. The fact that any coupling upper-bounds the optimal transport cost is classical, not a new Wasserstein result. The Pythia-specific development is an auditable experimental interface for producing the otherwise hidden near-replicate penalty.

## Why coupling quality matters

For scalar laws, sharing the same quantile rank across coordinates produces the monotone coupling, which minimizes expected absolute transport cost. Independent coordinate draws are also a valid coupling, but they include ordinary irreducible noise in `|Y_i-Y_j|` and can be far above W1.

In 100 fixed-design normal experiments with 1,000 environments each:

| Quantity | Result |
| --- | ---: |
| Numerical clipped mean W1 | 4.426 |
| Median common-quantile upper cap | 5.135 |
| Median independent-coupling upper cap | 6.588 |
| Cap misses, either route | 0/200 |
| Common-cap selected concentration | empirical Bernstein, 100/100 |

Common rank/quantile coupling is a strong design choice. In observational or counterfactual applications, rank preservation is an assumption and can be implausible; it must not be inferred merely because it makes the cap smaller. Here it is a controlled synthetic environment generator whose marginals are known by construction.

## Fresh ten-mechanism OOD audit

`run_coupled_environment_ood.py` froze:

- the existing ten near-replicate duration mechanisms;
- 800 evaluation entities;
- seeds 101001--101005;
- tuple sizes 2, 4, and 8;
- three score caps for each of MAE, Q90, and MSE;
- outcome clip `[0,300]`;
- 5,000 independent calibration environments;
- a 2.5% cap budget split over tuple sizes and internal concentration routes;
- a separate 2.5% budget split over all risk endpoints.

The common-quantile and product-coupled routes are separate 95% procedures. Their post-hoc maximum is not used.

| Route | Loss | Positive / 50 | Mechanisms with any positive run / 10 | Median lower/original-gold | Known-gold exceedances |
| --- | --- | ---: | ---: | ---: | ---: |
| Common quantile | MAE | **46** | **10** | 0.149 | 0 |
| Common quantile | Q90 | 0 | 0 | 0.000 | 0 |
| Common quantile | MSE | 0 | 0 | 0.000 | 0 |
| Independent product | MAE | 0 | 0 | 0.000 | 0 |
| Independent product | Q90 | 0 | 0 | 0.000 | 0 |
| Independent product | MSE | 0 | 0 | 0.000 | 0 |

Median caps across 50 datasets were:

| Tuple size | Common cap | Common empirical cost | Independent cap | Common tighter / 50 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8.753 | 8.286 | 16.138 | 50 |
| 4 | 15.595 | 14.868 | 25.271 | 50 |
| 8 | 20.831 | 19.999 | 30.936 | 50 |

`check_coupled_environment_population.py` reconstructed each synthetic marginal law and numerically integrated the common-quantile transport target on 5,000 quantiles. Across 150 dataset/tuple-size caps there were zero population-cap misses. The minimum cap-minus-numeric-W1 margin was 0.461 and the median margin 0.707. These numerical checks support implementation fidelity; the concentration theorem supplies the conditional guarantee.

## Interpretation

The result closes one earlier synthetic-oracle gap: the cap used by the MAE risk witness is now estimated from independent outcomes rather than read from the generator's `mu`, `scale`, or family parameters. It does **not** eliminate assumptions. It replaces an unobserved smoothness modulus with a concrete repeated-environment contract:

1. every environment coordinate must have the correct target-unit marginal law;
2. environment rows must be independent;
3. the joint coupling mechanism and any alignment must be registered;
4. the evaluation outcomes must be independent of cap calibration.

Ordinary patients in a retrospective table do not satisfy this contract. Rows have different covariates and do not provide repeated joint potential outcomes for the same target units. A fitted generative model could simulate such matrices, but then the certificate would inherit model-validity error; treating generated draws as if they were real independent environments would be invalid.

Potential legitimate sources include:

- a validated stochastic simulator with an external simulator-to-world discrepancy cap;
- repeated randomized operational scenarios sharing registered common shocks;
- physical/digital-twin experiments in which each target condition is replayed;
- prospectively collected repeated measurements whose coordinate marginals and dependence design are scientifically justified.

## Literature position

The coupling inequality is foundational optimal transport. Multi-environment ML commonly uses environments for invariance or causal generalization—for example [Yin, Wang, and Blei, JMLR 2024](https://jmlr.org/papers/v25/21-1028.html)—but that work estimates causal structure rather than an attainable-risk transport penalty. Rank-invariance and distributional-treatment-effect research emphasizes that a common-rank coupling is a substantive identifying restriction, not a free statistical fact. Pythia should describe repeated environments as a designed cap-audit resource, not as a new definition of optimal transport or a property of routine observational data.

## Remaining gap

MAE improved from a synthetic oracle-cap diagnostic to an outcome-calibrated proof of concept, but the required 5,000 joint environments are expensive and unavailable in current Mayo data. Q90 and MSE remain zero because their loss-specific transport penalties or score concentration dominate the witness. Next development should therefore separate two questions:

1. Can cap calibration be made sample-efficient through valid variance-adaptive or sequential upper confidence bounds?
2. Can pinball and MSE use sharper loss-specific discrepancies than the generic Lipschitz-times-W1 penalty?

No Mayo number or claim tier changes from this experiment.

## Follow-up: split learned alignment

A predeclared translation of each target unit preserves MAE, pinball, and MSE Bayes risk. `run_learned_alignment_environment_ood.py` therefore split 5,000 common-quantile environments into 2,500 alignment and 2,500 cap rows, estimated a per-unit median shift only on the first half, and calibrated the residual transport penalty only on the second. This avoids using the cap observations to learn their own alignment.

On fresh seeds 102001--102005, the aligned route produced positive MAE certificates in 50/50 runs across all ten mechanisms, with median lower/original-gold 0.536 and zero gold exceedances. Q90 was positive in 30/50 runs across six mechanisms (median ratio 0.179 over all runs; 0.365 among positive runs), also with zero exceedances. MSE remained zero in 50/50. A separate 150-cap numerical population check had zero misses and minimum cap-minus-target margin 0.413.

This is a valid and useful improvement to the transport route, but direct repeated-environment SAA is substantially tighter when complete repeated outcome matrices are available; see `REPEATED_ENVIRONMENT_RISK_CERTIFICATES_20261011.md`.

## Files

- Implementation: `coupled_environment_w1.py`
- Known-law check: `check_coupled_environment_w1.py`
- OOD runner: `run_coupled_environment_ood.py`
- Population diagnostic: `check_coupled_environment_population.py`
- Frozen rows: `coupled_environment_ood_results.jsonl`
- Split alignment runner: `run_learned_alignment_environment_ood.py`
- Split alignment population check: `check_learned_alignment_population.py`
- Split alignment rows: `learned_alignment_environment_ood_results.jsonl`

All stored rows are synthetic and contain no patient-derived data.
