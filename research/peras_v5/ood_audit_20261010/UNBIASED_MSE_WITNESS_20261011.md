# An unbiased repeated-environment MSE witness

## Development result

The original repeated-environment MSE route used the empirical optimum within each environment block. For block size `k`, that score is the sample variance with divisor `k`, whose expectation is only `(k-1)/k` of the conditional variance. The optimism is valid for a lower bound but unnecessarily loses `1/k` of the target before concentration.

For squared loss, the fixed-unit Bayes risk is exactly the conditional variance. Pythia now adds the Bessel-corrected within-unit sample variance as a second witness. It is unbiased, bounded after outcome clipping, requires no predictor, and can be selected jointly with the empirical-optimum witness under one alpha ledger.

## Theorem

For registered unit `i`, let `Y_1i,...,Y_ki` be independent draws from its clipped conditional outcome law, with support width `D`. Define

```text
U_i = (k-1)^-1 sum_e (Y_ei - mean_e Y_ei)^2.
```

Then

```text
E[U_i] = Var(Y_i) = inf_a E[(Y_i-a)^2].
```

For a block score `U = n^-1 sum_i U_i`, arbitrary dependence among units within each environment row is allowed and

```text
E[U] = n^-1 sum_i Var(Y_i) = R*_MSE.
```

By Popoviciu's inequality and the Bessel correction,

```text
0 <= U <= D^2 k / (4(k-1)).
```

For any registered cap `c`, `E[min(U,c)] <= R*_MSE`. Therefore any valid lower confidence bound for the mean of the independent clipped block scores is a lower confidence bound for the fixed-design MSE Bayes risk. The implementation routes Hoeffding, empirical Bernstein, and Bentkus across registered block sizes. When both MSE witness families are enabled, the family error is divided across every witness/block-size/concentration candidate before selection.

This is a loss-specific sharpening. Unbiased sample variance and Popoviciu's inequality are classical; the improvement is their integration into the Pythia attainable-risk router. It should not be described as a new variance theorem.

## Analytic check

On 100 independently generated Bernoulli repeated-environment problems:

- the raw unbiased score had median relative error 0.00283 against the analytic variance target;
- the selection-safe router selected the unbiased witness in 100/100 MSE cases;
- no lower endpoint exceeded the analytic target;
- the joint lower/upper interval checks had no contradictions.

## Fresh ten-mechanism audit

`run_mse_unbiased_ood.py` froze seeds 106001--106005 only after the implementation and analytic check were complete. It evaluated ten heterogeneous duration/noise mechanisms, 800 fixed units, 5,000 independent product-coupled environments, block sizes 2/4/8/16/32, and the same registered MSE score cap of 1,000.

| Result | Legacy empirical optimum | Unbiased variance | Selection-safe combined router |
| --- | ---: | ---: | ---: |
| Positive bounds | 50/50 | 50/50 | 50/50 |
| Median bound / known floor | 0.8204 | **0.9701** | **0.9686** |
| Strict wins over legacy | -- | **50/50** | **50/50** |
| Known-floor exceedances | 0 | 0 | 0 |
| Combined route chose unbiased witness | -- | -- | 50/50 |

The median improvement was 0.1385 of the true floor. Aggregating repetitions within each mechanism, the unbiased route improved all 10/10 mechanisms; the exact one-sided mechanism-level sign-test value is `2^-10 = 0.0009766`. Treating all 50 independently seeded runs as the unit gives `2^-50`, but the mechanism-level analysis is the more conservative primary comparison.

| Mechanism | Legacy mean bound/floor | Unbiased mean bound/floor | Gain/floor |
| ---: | ---: | ---: | ---: |
| 1 | 0.8064 | **0.9794** | 0.1730 |
| 2 | 0.8360 | **0.9836** | 0.1476 |
| 3 | 0.8424 | **0.9756** | 0.1332 |
| 4 | 0.8620 | **0.9850** | 0.1230 |
| 5 | 0.8160 | **0.9638** | 0.1478 |
| 6 | 0.8251 | **0.9541** | 0.1290 |
| 7 | 0.7680 | **0.9449** | 0.1770 |
| 8 | 0.8611 | **0.9822** | 0.1211 |
| 9 | 0.7005 | **0.7945** | 0.0940 |
| 10 | 0.7868 | **0.9321** | 0.1453 |

Mechanism 9 remains the hardest because the registered score cap truncates more of its high-variance blocks. Increasing the cap after inspecting these outcomes would be invalid. A future cap portfolio must be preregistered and charged in the same selection family.

## Scope

The result strengthens MSE only under the repeated-environment contract. It does not repair the A-fold orthogonality failure of the original one-outcome MSE construction and must not be used to relabel that route as proved. It also does not transfer to MAE or pinball loss: their empirical optimal risks do not have the sample-variance unbiasedness identity.

## Files

- Core: `repeated_environment_saa.py`
- Analytic check: `check_repeated_environment_saa.py`
- Fresh runner: `run_mse_unbiased_ood.py`
- Frozen rows: `mse_unbiased_ood_results.jsonl`

All audit rows are synthetic.
