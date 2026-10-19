# Retrospective Mayo family/neighborhood lattice diagnostic

Status: **exploratory**, aggregate-only, no model training.

The diagnostic used one earliest case for each of 9,300 patients; 8,210 had a
strictly pre-cutoff note bundle.  The nine-node simultaneous registry contained
global, procedure-family, procedure, and intact/within-procedure-permuted/
globally-permuted metadata and hashed bag-profile branches.

The code explicitly sets `iid_rows_attested=False` and
`conditional_exchangeability_attested=False`.  Exploratory overrides permit
numerical diagnostics but remove finite-sample coverage and valid-p-value
claims.  The working `[0,900]` minute cap is not an independently audited
population support.

## Aggregate lattice

| Representation | Cells | Empirical SAA floor | Exploratory lower endpoint |
|---|---:|---:|---:|
| global | 1 | 83.130 | 68.092 |
| procedure family | 8 | 59.501 | 44.463 |
| procedure | 139 | 45.663 | 30.625 |
| procedure + intact note metadata | 307 | 44.692 | 29.653 |
| procedure + intact hashed bag profile | 463 | 44.086 | 29.048 |
| procedure + within-procedure-permuted bag | 463 | 44.423 | 29.385 |
| procedure + globally permuted bag | 501 | 44.002 | 28.964 |

The decreasing empirical score under finer cells includes optimization bias;
it is not by itself evidence that notes lower population Bayes risk.

## Conditional signal diagnostic

The finite-refinement statistic compares the observed empirical SAA reduction
with 499 note-profile permutations within the frozen coarse cells.

| Registered increment | Observed SAA reduction | Null 99th percentile | Monte Carlo tail | Two-test adjusted |
|---|---:|---:|---:|---:|
| note metadata beyond procedure | 0.9715 min | 0.8882 | 0.002 | 0.004 |
| hashed bag beyond procedure + metadata | 0.6057 min | 0.7388 | 0.372 | 0.744 |

Thus note availability/volume carries an unusual amount of procedure-
conditional retrospective signal in this frozen profile, while the generic
hashed lexical sketch adds no detectable signal beyond metadata.  Because
clinical exchangeability is not attested, `0.002` is an **exploratory tail
area**, not a confirmatory p-value.  Note volume may also encode care-process
or timing effects and is not automatically deployable.

## Metamorphic checks

- Shuffling every note's token order left the bag profile exactly unchanged.
- Appending `X` repeated in proportion to the actual target also left the
  canonical profile unchanged when the registered parser removed pure-X noise.
- Exposing that target-coded X count to the representation was refused for
  missing outcome-blindness.

These distinguish a raw note channel from the information actually consumed
by a representation.  A target can exist in discarded text without changing
the parsed information set; if the parser consumes it, certification must
refuse.

## Provenance

- Tabular input SHA-256:
  `91c9e997dd0b623ee806c8267ce1fac67131e2170faafc0ded6a15e03a62f0f5`
- Note-bundle input SHA-256:
  `256be4a2e77153c9123d660da9799562d21d82b0b42c7f8db95d7da22c668d0e`
- Aggregate output SHA-256:
  `550bcda8f2fd3b5609bc97ab82acdb66aeb7025ef4758fd9bd4ef0838e0dbeef`

No note text, patient identifier, case identifier or row-level result is stored
in the repository artifact.
