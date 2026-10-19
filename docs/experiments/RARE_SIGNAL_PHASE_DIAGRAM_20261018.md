# Registered-support rare-signal phase diagram

Status: fresh synthetic development result, known conditional laws  
Command: `python3 research/peras_v5/nli_20261017/run_rare_phase_diagram.py 6000 5`

The comparison holds the table cells and binary concept fixed. The generic
route receives the concept as a one-dimensional geometry and greedily matches
nearby rows. The support route receives the same concept as a frozen finite
support label and uses the Helmert residual witness with heterogeneous-support
Chernoff concentration. Neither route trains a predictor.

| rare prevalence | logit signal | true residual Brier | generic positives / 5 | support positives / 5 | generic median truth fraction | support median truth fraction | support wins / 5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 1.5 | 0.00173 | 0 | 4 | 0.000 | 0.120 | 4 |
| 0.01 | 3.0 | 0.00415 | 0 | 5 | 0.000 | 0.378 | 5 |
| 0.02 | 1.5 | 0.00384 | 0 | 5 | 0.000 | 0.448 | 5 |
| 0.02 | 3.0 | 0.00819 | 0 | 5 | 0.000 | 0.542 | 5 |
| 0.05 | 1.5 | 0.00867 | 0 | 5 | 0.000 | 0.569 | 5 |
| 0.05 | 3.0 | 0.02105 | 0 | 5 | 0.000 | 0.628 | 5 |
| 0.10 | 1.5 | 0.01652 | 0 | 5 | 0.000 | 0.662 | 5 |
| 0.10 | 3.0 | 0.03830 | 2 | 5 | 0.000 | 0.755 | 5 |
| 0.20 | 1.5 | 0.02943 | 0 | 5 | 0.000 | 0.719 | 5 |
| 0.20 | 3.0 | 0.06826 | 5 | 5 | 0.324 | 0.785 | 5 |

Totals:

- support strict wins: **49/50**, losses: **0**, tie: **1**;
- support truth violations: **0/50**;
- at 1% prevalence: support positive **9/10**, generic positive **0/10**;
- at 2%--20% prevalence: support positive **40/40**.

The descriptive sign probability over the 49 non-ties is `2^-49`, but the
suite is developmental and this is not a preregistered confirmatory p-value.

## Interpretation

The generic geometry does not know that a rare state should be pooled across
all its occurrences. A registered support label supplies exactly that
outcome-blind structural fact. Pythia then tests the low-dimensional
support-by-outcome interaction directly instead of spending evidence on a
large matching graph. This is an information-contract distinction, not an
algorithm secretly discovering a target-correlated token.

A support absent from the registry cannot receive this gain. Supports below
`min_support` return `insufficient_resolution`, and all registered candidates,
outcome axes, and signs are charged to the alpha ledger.
