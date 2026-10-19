# Curvature-penalty confirmation results

Protocol: `CURVATURE_CONFIRMATION_REGISTRATION_20261014.md`.
Raw rows: `curvature_confirmation_results.jsonl`.
SHA-256: `560a5af465bf9869a738c565f138a647aaefffa1a930474baf8f365dafddb010`.

There are 400 rows: ten mechanisms, ten untouched seeds, two outcome-blind
geometry strategies, and two losses.  Score caps and geometry fits use only
the learning half; the certification half is opened once.

## Primary results

| Loss | Geometry | Rows | Barycentric median bound / gold | Curvature median bound / gold | Curvature strict wins | Mechanism-median wins | Curvature violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAE | learned key | 100 | 0.5018 | **0.8356** | 100 | 10/10 (`p=0.00098`) | 0 |
| MAE | raw + learned alignment | 100 | 0.5244 | **0.8403** | 100 | 10/10 (`p=0.00098`) | 0 |
| Q90 | learned key | 100 | 0.0000 | **0.5979** | 79 (21 ties) | 8/10 (`p=0.0547`) | 0 |
| Q90 | raw + learned alignment | 100 | 0.0000 | **0.6285** | 97 (3 ties) | 10/10 (`p=0.00098`) | 0 |

The sign-test p-values use mechanisms—not seed rows—as the unit.  They are
one-sided descriptive confirmation tests fixed in the registration.

Both routes had zero known-floor exceedances in all 400 rows.  This simulation
record is consistent with validity but does not replace the theorem or prove a
95% coverage rate from only 400 correlated loss/configuration rows.

## Mechanism view

The table pools the two geometry strategies and reports the median over their
20 rows per mechanism.

| Mechanism | MAE bary | MAE curvature | Q90 bary | Q90 curvature |
|---|---:|---:|---:|---:|
| discrete cells | 0.774 | **0.856** | 0.553 | **0.730** |
| Friedman | 0.503 | **0.852** | 0.000 | **0.596** |
| heteroscedastic | 0.409 | **0.732** | 0.000 | **0.009** |
| high frequency | 0.550 | **0.852** | 0.000 | **0.648** |
| high SNR | 0.250 | **0.743** | **0.000** | **0.000** |
| linear | 0.525 | **0.851** | 0.000 | **0.627** |
| low SNR | 0.586 | **0.852** | 0.044 | **0.673** |
| lognormal skew | 0.445 | **0.620** | 0.000 | **0.116** |
| sparse 20D | 0.565 | **0.862** | 0.000 | **0.671** |
| t3 | 0.462 | **0.803** | 0.000 | **0.446** |

## Interpretation

The confirmatory result supports the diagnosis that first-order transfer—not
concentration—was the dominant MAE bottleneck.  It also repairs Q90 on most
mechanisms.  It does **not** solve extreme Q90 universally: high-SNR remains
zero, heteroscedastic Q90 is near zero, and skew Q90 remains weak.  Those are
now localized to the global curvature envelope and finite-block pinball
witness/cap rather than hidden by the old penalty.

The comparison is deliberately internal: it tests a theorem refinement while
holding witness, geometry, concentration, data, and alpha spending fixed.  It
is not evidence that Pythia is universally superior to every published method.

All action dispersions and density envelopes in this confirmation come from
the known simulator laws.  A deployable non-oracle route needs the independent
repeated-environment calibration or an external declaration.  No Mayo number
changes.

