# Audit of the declared-modulus attachment

## Decision

The attached theorem is a valid and useful **conditional sensitivity route**.
It does not identify the residual-law discrepancy from one observational
outcome per unit, and its synthetic evaluation at the true discrepancy is an
oracle theorem check.  It therefore strengthens Pythia, but it does not turn
the Mayo result into a proved certificate and it is not a train-free estimator
of the discrepancy.

The production package now exposes the safe feature-modulus bridge through
`feature_barycentric_dispersion` and `declared_feature_modulus_profile`.  It
does not import the attachment's research scripts.  In particular, it never
fits the HistGradientBoosting location map: an external frozen shift may be
used only with provenance and independence from evaluation outcomes.

## What was independently reproduced from the supplied rows

| Audit | Result |
| --- | --- |
| Declared-modulus, no shift, MAE | 50/50 positive; median bound/gold 0.1887; 0 oracle-gold exceedances |
| Declared-modulus, independent shift, MAE | 50/50 positive; median 0.4362; 0 exceedances |
| Declared-modulus, independent shift, MSE | 50/50 positive; median 0.4685; 0 exceedances |
| Declared-modulus, independent shift, Q90 | 32/50 positive; median 0.0850; 0 exceedances |
| Fresh location-only replication, MAE/MSE | medians 0.4482/0.4843; 0 exceedances |
| Location-scale negative control | tighter in 0/50 for MAE, Q90, and MSE |
| Exact environment-replicate SAA vs transport | tighter in 50/50 for MAE, Q90, and MSE; 0 exceedances |

Supplied artifact hashes:

- `declared_modulus_ood_results.jsonl`:
  `757bb6481c80ec8a86614b2f47c71c310bad8ee12e5ed94f5d88fa69eed8d917`
- `location_scale_alignment_results.jsonl`:
  `7b6fa314ecbd308ce1862b2f1b45b8befcde97738b91e63cbbabee347ee3ab2c`
- `env_replicate_saa_results.json`:
  `8a83303c6205dc9f7c02ecea51a0ecd57ac1ae2c7c2e0696fb2b92e57aaf1f2c`

These checks verify arithmetic and internal consistency of supplied result
rows.  They do not independently regenerate the rows from a sealed execution
environment.

## The accepted theorem interface

For outcome-blind disjoint block `t`, frozen feature metric `d`, and residual
laws `Q_j`, define

```
G_t = k^-2 sum_ij d(z_i,z_j)
H_t = k^-1 sum_j (k^-1 sum_i d(z_i,z_j))^2.
```

If the declared within-block modulus is valid, the barycentric discrepancy is
at most `Lambda G` for MAE/pinball and `Lambda_2^2 H` for MSE.  Thus one
simultaneous lower score event gives the entire curve

```
MAE:      retained * max(0, LCB - Lambda G)
pinball:  retained * max(0, LCB - 0.5 Lambda G)
MSE:      retained * max(0, LCB - Lambda_2^2 H).
```

Evaluating many declared values on this curve spends no additional alpha.
Selecting among different block/score constructions is safe only when their
lower endpoints were made simultaneous before selection.

The repository's newer direct-`P(k)` route is more general: a declaration of
the aggregate barycentric discrepancy need not factor as `Lambda G`.  The
feature-modulus route is therefore an auditable sufficient-condition adapter,
not a replacement.

## Required claim corrections

1. “Feature-only certificate” is too strong. `G` and `H` are feature-only;
   `Lambda` is an external structural declaration.  The result is conditional.
2. The oracle `Lambda_bar=P/G` used in the synthetic tables validates the
   theorem but is unavailable in ordinary Mayo data.
3. The independently trained location map is useful and legitimate as a
   frozen external resource, but it is model training.  Pythia itself must only
   consume the frozen map and its provenance.
4. A twin-implied Mayo discrepancy is calibrated/model-based, not proof of the
   real cohort's discrepancy.
5. A zero feature distance is not evidence of equal residual laws.  It gives a
   zero penalty only if the declared modulus assumption truly holds.
6. Q90 remains the weakest route.  The attachment does not resolve the
   small-block pinball optimism/concentration problem.

## Consequence for Mayo

The attachment adds a useful reportable curve:

> MAE is at least `B(Lambda)` if the frozen metric's residual-law W1 modulus
> does not exceed `Lambda` over the registered blocks.

It does not supply a defensible Mayo value of `Lambda`.  Until an external
simulator, repeated-outcome design, expert-elicited cap with falsification, or
prospective resource supplies that value, the Mayo endpoint remains
`conditional`/`sensitivity_only` rather than certified.

## Next research gate

The most valuable next step is not another oracle-modulus synthetic win.  It
is a non-oracle protocol for declaring and stress-testing `Lambda` on held-out
environments, plus a cluster/time-dependent score endpoint.  Separately, Q90
needs a multi-point barycentric witness whose selection and concentration are
accounted under one frozen ledger.
