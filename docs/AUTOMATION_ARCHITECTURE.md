# Dataset-agnostic automation architecture

Pythia is the certification engine. Mayo is one study configuration and a set of
reproduction adapters; it must never become an implicit default.

## Automated study flow

1. **Declare** — a YAML study assigns the row ID, target, entity, time, loss, and split
   policy. Dataset-specific column names stop at this boundary.
2. **Prepare** — Horos validates the schema, hashes inputs, creates a versioned fold
   manifest, and performs leakage/provenance checks.
3. **Represent** — registered representation adapters build information sets while emitting
   feature names, availability rules, and source hashes.
4. **Cross-fit** — the generic OOF runner invokes registered model adapters and refuses
   incomplete predictions or entities crossing folds.
5. **Certify** — Pythia consumes only the task card, aligned OOF matrix, representation
   provenance, and explicit guard. It does not know Mayo column names or model classes.
6. **Compare** — the ladder compares aligned representations, estimates achieved movement
   and floor movement, and runs clustered uncertainty calculations.
7. **Gate and report** — claim tier, assumptions, hashes, falsification checks, and refusal
   reasons are serialized. Metis may automate search only after these gates are stable.

## Extension interfaces

A new dataset should require:

- one study YAML;
- optional representation adapters;
- optional model adapters implementing fold-local fit/predict;
- a loss guard only when the registered loss requires one.

The grouped temporal manifest and crossfit executor are already generic. The tuned Mayo
XGBoost code is a historical model adapter, not part of the mathematical core. The next
implementation step is a registry for model and representation adapters followed by a single
`pythia run STUDY.yaml` command that executes prepare → cross-fit → certify.

## Current Mayo fixture

`mayo.study.yaml` exercises the generic preparation path. It reproduces 10,882 development
rows, 2,839 temporal holdout rows, and 474 excluded earlier rows from future-holdout patients.
The first regenerated model candidate has development OOF MAE `33.2749753373`; it passes exact
row, fold, and score auditing. It is not relabeled as the unsupported 31.2623 incumbent.

## Generic certificate transport

`pythia certify CONFIG` now accepts a dataset-independent `certificate:` YAML
request backed by a hash-pinned NPZ artifact. Supported input kinds are
`fold_matrix` and `pair_matrix`; supported generic guards are
`dependence_finite_variance`, `dependence_bounded_range`, and `replicate`.
The loader uses `allow_pickle=False`, validates required arrays and any declared
SHA-256, and never invokes Horos model code.

Minimal replicate request:

```yaml
certificate:
  task:
    target: duration
    loss: mae
    operational_unit: minutes
  representation:
    name: frozen-preoperative-metric
    version: "1"
  input:
    kind: pair_matrix
    artifact: pairs.npz
    sha256: REPLACE_WITH_FROZEN_HASH
    meta:
      pairing_outcome_blind: true
      conditionally_independent_entities: true
  guard:
    kind: replicate
    version: replicate-v1
    lipschitz_bound: 400
    distance_bound: 1
    outcome_difference_bound: 900
    calibration_population: independent historical smoothness audit
    source_hash: REPLACE_WITH_GUARD_HASH
    concentration: auto
```

The NPZ must contain `y_left`, `y_right`, `distances`, `left_entities`,
`right_entities`, and `pair_blocks`. A constructor records that its code does
not accept outcomes, but deliberately leaves `pairing_outcome_blind`,
`conditionally_independent_entities`, `independent_blocks`, and
`iid_randomized_blocks` false. Only a separate provenance audit may set those
claims true. With `concentration: auto`, Pythia selects pair-level conditional
empirical Bernstein when entity independence is audited, downgrades to
conditional block empirical Bernstein when only independent blocks are
audited, and refuses when neither is available. Invalid contracts return a
structured refusal and CLI exit code 2.
