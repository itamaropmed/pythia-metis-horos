# Fixed-policy information-value confirmation results

Date: 2026-10-14

The frozen protocol in `FIXED_POLICY_VALUE_CONFIRMATION_REGISTRATION.md` used
50 fresh seeds for each of ten known-law mechanisms and 8,192 entities per
run.  Pythia received losses from a predeclared analytic refined-information
policy and did not fit a predictor.

## Primary endpoints

- Analytic information-value exceedances: **0 / 500**.
- Positive information-value lower endpoints: **500 / 500**.
- Strict wins over matched global-range/Hoeffding baseline: **500 / 500**.
- Mechanism-median wins: **10 / 10**; two-sided sign test
  `p = 0.001953125`.
- Median bound / true information value: **0.642925**.
- Matched baseline median bound / truth: **0.454621**.
- Uniform-noise positive endpoints: **50 / 50**.

| Mechanism | True information value | New median bound / gold | Baseline median / gold |
|---|---:|---:|---:|
| bimodal | 0.419459 | **0.525920** | 0.374677 |
| discrete mixture | 0.504141 | **0.703800** | 0.652558 |
| Gaussian | 0.424869 | **0.647654** | 0.511702 |
| heteroscedastic | 0.441771 | **0.616813** | 0.250163 |
| high SNR | 0.063730 | **0.647654** | 0.031393 |
| lognormal | 0.606994 | **0.756830** | 0.614301 |
| low SNR | 0.849738 | **0.647654** | 0.554082 |
| t3 | 0.883555 | **0.777013** | 0.695461 |
| triangular | 0.110747 | **0.546280** | 0.170709 |
| uniform | 0.064768 | **0.173535** | 0.000000 |

## What is certified

For coarse risk `R_C`, refined Bayes risk `R_F`, a simultaneous coarse lower
endpoint `L_C`, and an upper endpoint `U_pi` on the risk of a policy measurable
from the refined information,

```text
R_C - R_F >= R_C - R(pi) >= L_C - U_pi.
```

The implementation allocates alpha before inspecting outcomes.  The
confirmation predeclared the dKelly upper-mean route after a separate
resolution study; it was not selected from confirmation outcomes.

The result does not require Pythia to train a model, but it does require a
genuinely frozen, pre-outcome refined-information policy evaluated under the
registered sampling contract.  A weak policy gives a valid but possibly zero
endpoint.  Ordinary reused OOF predictions or a retrospectively chosen Mayo
winner do not automatically satisfy this contract.

Raw artifact SHA-256:
`f920b16e0adf28a2862b2213f238693e591c790fd5632a8281114af9ffe20149`
