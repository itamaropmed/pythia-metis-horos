# Pythia local frozen-model risk certificates

Status: **implemented conditional theorem; development validation passed**  
Scope: bounded loss, finite outcome-blind hierarchy, independent entity clusters  
Not: an individual-error guarantee, a trained risk model, or a prospective Mayo claim

## The gap this closes

The earlier Pythia information-floor certificate described an information set
globally. Metis attached an accurate but exploratory risk forecast to an
arriving case. Those are different objects. A global attainable-error floor is
not automatically a certificate for one model on one local population.

Pythia now exposes a third object: a simultaneous interval for the expected
loss of each frozen model in each predeclared, outcome-blind cell. An arriving
case selects the finest cell with enough independent-patient support. Because
all cells, hierarchy levels, and models share one familywise alpha ledger, this
post-calibration routing does not invalidate the intervals.

## Fixed-design theorem

Let model `m`, hierarchy node `c`, and entity clusters be fixed without reading
the certified outcomes. Cluster `g` contributes `n_gc` cases to node `c` and
has case losses in `[0,M]`. Define

`X_gmc = sum(loss_i : i belongs to g,c) / n_c`.

Conditional on the fixed memberships and row counts,
`0 <= X_gmc <= n_gc M / n_c`, clusters are independent, and
`sum_g E[X_gmc]` is the case-weighted expected loss in the fixed node. Pythia
inverts its heterogeneous-support Chernoff envelope for the lower endpoint and
applies the same operation to the complemented contributions for the upper
endpoint.

If there are `K` model-node pairs, each tail receives `alpha/(2K)`. A union
bound therefore covers every interval simultaneously with probability at least
`1-alpha`. Any routing function that only reads arrival-time covariates and
these already-frozen intervals inherits the same event.

Repeated rows from one patient are never treated as independent. They enter a
single cluster contribution with the exact outcome-blind support
`n_gc M / n_c`.

## Non-arbitrary categorical risk

The routed local and global intervals are part of the same simultaneous event.
Pythia reports:

- `certified_higher_than_global` if the local lower endpoint exceeds the global
  upper endpoint;
- `certified_lower_than_global` if the local upper endpoint is below the global
  lower endpoint;
- `local_shift_unresolved` when a supported local interval overlaps global;
- `global_only` when the hierarchy must back off to the global node;
- `refused` when assumptions or support fail.

Thus unresolved evidence is not labeled “zero information.” The numerical
output is the local estimate and simultaneous interval; the category expresses
only what the intervals logically certify.

## Fail-closed contract

The implementation refuses unless the caller attests:

1. the complete finite hierarchy was fixed without outcomes;
2. entity clusters are mutually independent;
3. a hard almost-sure loss cap is valid;
4. new-case routing does not read calibration outcomes.

An observed cap violation is an error. A node with fewer than two entities gets
the trivial `[0,M]` interval and cannot be selected when `min_entities >= 2`.

For Mayo, W30 miss loss has a natural cap of one. Absolute error does not have a
natural finite cap, so Pythia must refuse certified MAE-local-risk intervals
unless an externally defensible cap or a separately justified truncation
estimand is declared.

## Separation from risk forecasting

This certificate is training-free. Horos may train a prediction model or an
exploratory conditional-risk forecaster; Pythia consumes only frozen losses,
hierarchy labels, entity IDs, and attestations. Correlation between a risk
forecast and realized loss is a useful resolution diagnostic, but it is not a
Pythia objective and has no default threshold.

Implementation: `pythia/core/local_model_risk.py`  
Synthetic audit: `scripts/run_pythia_local_risk_validation.py`  
Mayo replay: `scripts/run_mayo_pythia_local_w30.py`

## Remaining proof and product gates

1. Independent theorem/proof and numerical-optimizer audit.
2. Preregistered untouched Monte Carlo suite with Wilson coverage reporting.
3. Explicit covariate-shift/transport layer; development-cell coverage alone
   does not guarantee a future temporal cohort.
4. Confidence-sequence or e-process extension if Metis adapts the hierarchy
   after deployment.
5. Decide whether the product should report cell-average risk only or add a
   separately calibrated individual predictive distribution. Do not conflate
   the two.
