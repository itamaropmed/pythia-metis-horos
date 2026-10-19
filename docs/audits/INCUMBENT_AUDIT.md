# Mayo incumbent provenance audit

## Finding

The repository does not currently contain evidence that a case-duration model achieved
held-out MAE `31.2623`.

The only matching value located in the migration source is
`31.261751972973382` in:

`output/mayo_cure_sparse_calibrated_10k/metrics_summary.json`

Its JSON path is:

`selected_residual_correctors.cure_residual_stack.selection_table[3].oof_residual_mae`

The field measures OOF error for predicting the structured model's **residuals** during
corrector selection. It is not the MAE of final case-duration predictions and it is not a
held-out endpoint. The same run's best reported held-out duration result is
`37.40069949433148`; its structured-only held-out result is `37.6108141588843`.

Consequently, `31.2623` must not be called the operational incumbent or supplied to Pythia
until a distinct artifact proves that interpretation. This corrects the earlier contextual
label without deleting the historical record.

## Acceptance contract

An incumbent OOF artifact must contain, in canonical development order:

- `case_durable_id`;
- `patient_durable_id`;
- `fold_id` matching `canonical_folds.parquet`;
- `case_actual_duration_time`;
- one clearly named final-duration prediction column.

`pythia audit-incumbent --manifest ...` now checks the score, exact ordered cohort, and exact
fold assignments. A scalar metric or held-out-only prediction file cannot pass.

## Canonical historical split

The reconstructed historical contract is
`grouped-temporal-holdout+groupkfold-v1`: a patient-disjoint temporal outer holdout,
followed by five patient-grouped (not forward-chaining) development folds. Changing the
inner split to temporal forward chaining is a new validation design and must not be used to
claim reproduction of the historical score.
