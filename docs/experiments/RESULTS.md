# Fathom on Mayo HRS case duration (minutes)

n = 12,993 cases (baseline v2 exclusions, 2023-01-01 burn-in), 10,532 patients.
Cross-fitting: patient-grouped 5-fold, two distinct partitions, 6 out-of-fold
predictors (HGB absolute_error, XGBoost reg:absoluteerror, ExtraTrees).
Guard envelope frozen from the 496-pair shape library.

| arm | p | L (best OOF MAE) | M+ | rho | B (unguarded) | **B\* (shape guard)** | B_sf | B_sqrt2 | **E (estimate)** | temporal holdout MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| tabular only | 296 | 33.32 | 10.96 | 0.667 | 31.47 | **26.52** | 24.40 | 20.86 | **32.75** | 32.28 [30.89, 33.50] |
| + 11 note-derived features | 307 | 32.87 | 10.67 | 0.669 | 31.09 | **26.22** | 24.13 | 20.65 | **32.31** | 31.67 [30.36, 32.85] |
| + note text (TF-IDF/SVD-100) | 409 | 33.82 | 10.41 | 0.669 | 32.18 | 27.21 | 25.07 | 21.53 | 33.30 | 32.01 [30.69, 33.22] |

All quantities in minutes. Guard qhat = 1.1611 on all three arms.
