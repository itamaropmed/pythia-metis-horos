# Repeated-environment attainable-risk certificates

## Honest headline

Pythia can tightly certify the fixed-design Bayes risk for several losses **without training a predictive model** when the data contain independently repeated outcomes for every registered information state. This is the strongest multi-loss result in the current programme. It is mathematically valid under a clear repeated-environment contract and was strong on ten fresh synthetic mechanisms.

It is not a solution for ordinary one-outcome-per-patient data. Current Mayo data do not provide repeated independent outcomes for the same patient information state, so this result does not change the Mayo certificate. Simulated replicates from a learned generator cannot be treated as real environments unless a separate simulator-to-world discrepancy guarantee is supplied.

## Target and data contract

Fix target units `i = 1,...,n`, including each unit's complete registered pre-outcome information. Let `Q_i` be the conditional outcome law for unit `i`. Environment row `e` supplies

```text
(Y_e1,...,Y_en),
```

with coordinate marginal `Y_ei ~ Q_i`. Dependence among coordinates within an environment is arbitrary. Environment rows are independent and identically distributed.

The target is the fixed-design average clipped-outcome Bayes risk

```text
R* = n^-1 sum_i inf_a E_Qi loss(Y,a).
```

Because the target conditions on the full registered information of each unit, it is agnostic to how that information was engineered: tabular transformations, extracted concepts, and text representations are all allowed if they were fixed before outcomes and are genuinely part of the registered unit state. It does not transfer automatically to unregistered raw information, future populations, changed note pipelines, or outcomes outside the declared clip.

## Lower-certificate theorem

Partition the independent environment rows into blocks of size `k`. For block `b` and unit `i`, compute the empirical optimal risk

```text
V_bi = inf_a k^-1 sum_e loss(Y_ei,a),
S_b  = n^-1 sum_i V_bi.
```

Then

```text
E[S_b]
  = n^-1 sum_i E[inf_a empirical-risk_i(a)]
 <= n^-1 sum_i inf_a E[empirical-risk_i(a)]
  = R*.
```

Thus a one-sided lower confidence bound on `E[min(S_b,c)]`, for a registered cap `c`, is also a lower confidence bound on `R*`. Independent blocks provide the sampling units. The implementation computes Hoeffding, empirical-Bernstein, and Bentkus endpoints over a predeclared portfolio of block sizes and pays for every candidate through a finite alpha split before selecting the largest endpoint.

This is a sample-average-approximation optimism argument combined with bounded-mean concentration. Those ingredients are classical. The candidate Pythia contribution is their loss-routed, selection-safe use as an attainable-risk certificate with explicit provenance and refusal—not a claim to have invented `E inf <= inf E`.

## Upper endpoint and point estimate

An independent upper endpoint is obtained by splitting environment rows in half. The first half estimates one action per fixed unit; the second half evaluates those frozen actions. Since the risk of any action is at least the Bayes risk, a one-sided upper confidence endpoint for this held-out risk is an upper endpoint for `R*`.

The full-replicate empirical optimum is also returned as an exploratory plug-in estimate. It is not called certified because its finite-sample downward bias is not covered by a two-sided theorem. On the fresh synthetic registry it was nevertheless extremely accurate.

Estimating a scalar action for each already-registered unit is a risk functional calculation, not training a predictor that maps features to outcomes. No learned prediction rule is used by the lower certificate.

## Fresh ten-mechanism experiment

`run_repeated_environment_saa_ood.py` froze ten heterogeneous duration-like mechanisms, five new seeds per mechanism, 800 fixed target units, and 5,000 environment rows. Product-coupled and common-quantile environments had identical coordinate marginals and were audited separately.

### Location losses: lower certificates

| Loss | Product positive / 50 | Product median lower / true floor | Common median lower / true floor | Known-floor exceedances | Plug-in median relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAE | 50 | **0.859** | 0.811 | 0 | 0.0023 |
| MSE | 50 | **0.799** | 0.705 | 0 | 0.0142 |
| Q10 pinball | 50 | **0.798** | 0.746 | 0 | 0.0019 |
| Q50 pinball | 50 | **0.859** | 0.811 | 0 | 0.0023 |
| Q90 pinball | 50 | **0.794** | 0.732 | 0 | 0.0060 |

The product-coupled lower endpoint exceeded the common-coupled endpoint in all 250 matched dataset/loss comparisons. This is not because the target changed: within-row coupling changes the variance of the block-average score, while its expectation remains a lower witness for the same marginal-law target. Independent product coupling reduced across-unit score covariance in these mechanisms.

### MSE loss-specific sharpening

A subsequent fresh audit replaced the downward-biased empirical MSE optimum with the Bessel-corrected within-unit sample variance. Its expectation equals the fixed-design MSE Bayes risk exactly, while bounded clipping supplies a finite score range. Across seeds 106001--106005 it beat the former MSE witness in all 50/50 runs and all 10/10 mechanisms, raising median bound/known-floor from 0.8204 to 0.9701 with zero exceedances. The combined selection-safe router, which paid for both witness families, recovered 0.9686 and selected the unbiased route in 50/50 runs. See `UNBIASED_MSE_WITNESS_20261011.md`.

### Dependence-aware sampling-unit sharpening

When product independence across target coordinates is part of the registered environment design, block--unit witnesses are mutually independent and can be concentrated directly. An untouched, selection-safe row/coordinate router recovered median fractions 0.9723 (MAE), 0.9658 (MSE), 0.9428 (Q10), 0.9723 (Q50), and 0.9207 (Q90), with zero gold exceedances. It selected coordinate concentration for every MAE/pinball run and chose between coordinate and row MSE according to the interaction of variance and score truncation. A separate fresh window audit improved W15/W30/W60 median tightness from 0.643/0.356/0.019 to 0.840/0.773/0.497. See `DEPENDENCE_AWARE_SAMPLING_UNIT_20261011.md`.

### Two-sided intervals

`run_repeated_environment_intervals_ood.py` used fresh seeds 105001--105005 and disjoint 2.5% lower and upper error budgets.

| Loss | Median lower / floor | Median upper / floor | Median relative interval width | Lower misses | Upper misses | Contradictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE | 0.856 | 1.122 | 0.259 | 0 | 0 | 0 |
| MSE | 0.813 | 2.131 | 1.318 | 0 | 0 | 0 |
| Q10 | 0.795 | 1.467 | 0.664 | 0 | 0 | 0 |
| Q50 | 0.856 | 1.122 | 0.259 | 0 | 0 | 0 |
| Q90 | 0.794 | 1.431 | 0.638 | 0 | 0 | 0 |

Across 250 intervals there were no numerical gold violations and no lower-greater-than-upper contradictions. The MSE upper endpoint remains loose because squared loss pays the full clipped range; a tighter upper requires a valid tail or variance condition, not post-hoc winsorization.

### Window-miss losses

`run_repeated_environment_windows_ood.py` used fresh seeds 104001--104005. For window radius `h`, the block score computes one minus the largest fraction of the `k` observations contained in any interval of width `2h`; it is the empirical optimal window-miss risk and obeys the same `E inf <= inf E` theorem.

| Loss | Positive / 50 | Median lower / true floor | Known-floor exceedances | Plug-in median relative error |
| --- | ---: | ---: | ---: | ---: |
| W15 miss | 50 | **0.644** | 0 | 0.0069 |
| W30 miss | 50 | **0.356** | 0 | 0.0116 |
| W60 miss | 40 | 0.016 | 0 | 0.0723 |

W60 is often near zero in this registry, so absolute rather than relative tightness should be inspected before interpreting its lower ratio.

## Analytic and numerical checks

`check_repeated_environment_saa.py` includes:

- 100 Bernoulli-law repetitions with analytic MAE, MSE, quantile, and window targets;
- zero lower misses, zero upper misses, and zero interval contradictions;
- positive lower bounds in all tested nondegenerate Bernoulli cases;
- median exploratory plug-in relative error 0.00279;
- 60 additional uniform-window cases with zero lower-bound violations;
- a structured refusal when repeated-environment provenance is absent.

These are implementation checks, not a proof by simulation. The proof is the expectation inequality plus simultaneous concentration; the experiments measure usefulness and catch implementation errors.

## Relation to the coupled-Wasserstein route

The earlier coupled-environment method first upper-bounds a within-block Wasserstein penalty and subtracts it from a geometric witness. Under common-quantile environments it recovered positive MAE bounds in 46/50 runs, but no positive Q90 or MSE bounds. Sample-split per-unit median alignment improved that route to:

| Loss | Positive / 50 | Mechanisms reached / 10 | Median lower / true floor | Gold exceedances |
| --- | ---: | ---: | ---: | ---: |
| MAE | 50 | 10 | 0.536 | 0 |
| Q90 | 30 | 6 | 0.179 over all runs | 0 |
| MSE | 0 | 0 | 0.000 | 0 |

Direct repeated-environment SAA is stronger whenever full coordinate replicates are available because it estimates the relevant loss functional directly and avoids paying an intermediate transport penalty. The alignment/cap route remains relevant when only partial paired conditions or externally bounded transport resources are available.

## What the result establishes

1. Under the repeated-environment contract, Pythia has a general finite-sample lower certificate for any bounded loss whose finite-block empirical optimum can be computed.
2. The same engine handles MAE, MSE, asymmetric quantile loss, and window-miss percentage without predictor training.
3. On the frozen ten-mechanism registry, it was positive on every location-loss run, typically recovered 79--86% of the known floor, and showed no observed validity failure.
4. An exploratory estimator was accurate to about 0.03--1.4% median relative error for the tested location losses.
5. The method is genuinely agnostic to feature engineering only for the same fixed registered information states represented by the repeated coordinates.

## What it does not establish

1. It does not provide a universal bound from ordinary iid rows containing one outcome each.
2. It does not certify arbitrary future feature engineering or unregistered raw clinical notes.
3. It does not prove superiority over every published Bayes-risk method; the current strongest comparison is internal matched construction and known truth.
4. It does not make synthetic generator draws equivalent to real repeated measurements.
5. It does not update the Mayo 28.85 calibrated result or make the Mayo study confirmatory.
6. It is not yet a complete Pythia backbone: population transfer, one-observation continuous-information certification, dependence/shift composition, external comparator fidelity, and automatic search accounting remain open.

## Research verdict

This result materially strengthens Pythia and supports a focused paper about **data-contract-aware attainable-risk certification**. The reusable insight is not a universal new definition of information; it is that different experimental resources expose different certified projections of irreducible risk, and a fail-closed compiler can route each resource to the strongest valid loss-specific witness.

The most defensible novelty direction is therefore:

> A compositional certificate system that declares the information target and available data resource, produces simultaneous lower/upper attainable-risk statements across losses, accounts for every adaptive choice, and refuses when the bridge from observed data to the requested information set is unidentifiable.

That is promising and plausibly paper-worthy after external baselines and a real repeated-measurement application. A claim that Pythia already bounds every possible model from arbitrary tabular/text data would be false.

## Files

- `repeated_environment_saa.py`: lower certificate, exploratory plug-in, and held-out upper endpoint.
- `check_repeated_environment_saa.py`: analytic-law, window, interval, and refusal checks.
- `run_repeated_environment_saa_ood.py`: location-loss OOD audit.
- `run_repeated_environment_windows_ood.py`: window-loss OOD audit.
- `run_repeated_environment_intervals_ood.py`: fresh two-sided audit.
- `repeated_environment_saa_ood_results.jsonl`: frozen location-loss rows.
- `repeated_environment_windows_ood_results.jsonl`: frozen window rows.
- `repeated_environment_intervals_ood_results.jsonl`: frozen interval rows.
- `run_mse_unbiased_ood.py`: fresh loss-specific MSE comparison.
- `mse_unbiased_ood_results.jsonl`: frozen MSE-sharpening rows.
- `run_dependence_aware_ood.py`: untouched selection-safe dependence router audit.
- `dependence_aware_ood_results.jsonl`: frozen router rows.
- `run_coordinate_windows_ood.py`: fresh window dependence audit.
- `coordinate_windows_ood_results.jsonl`: frozen window dependence rows.
- `run_learned_alignment_environment_ood.py`: split alignment experiment.
- `check_learned_alignment_population.py`: numerical population-cap check.

All stored rows are synthetic and contain no patient-derived data.
