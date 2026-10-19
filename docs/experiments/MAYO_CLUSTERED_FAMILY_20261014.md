# Repeated-patient Mayo family/neighborhood diagnostic

Status: **exploratory**, aggregate-only, no model training.

This analysis retains all 11,356 eligible cases from 9,300 patients instead of
discarding repeated cases.  There are 2,056 cases beyond one per patient, the
maximum observed cluster size is six, and the count-based effective cluster
size is 7,835.6.  Arbitrary dependence among one patient's cases is allowed by
the numerical cluster width.

Cluster independence, temporal stability, and the population validity of the
working `[0,900]` minute support are not attested.  Every endpoint below is
therefore exploratory.

## Simultaneous five-node family

The family alpha is 0.05, split equally across the five frozen nodes.

| Representation | Cells | Empirical SAA | Range-based cluster lower | Support-free block lower | Invalid row-IID lower |
|---|---:|---:|---:|---:|---:|
| Global | 1 | 83.434 | 68.005 | **77.398** | 70.618 |
| Procedure family | 8 | 61.777 | 46.349 | **51.765** | 48.961 |
| Procedure | 153 | 45.438 | 30.010 | **30.094** | 32.622 |
| Procedure + note metadata | 341 | 44.532 | **29.103** | 25.398 | 31.716 |
| Procedure + hashed bag | 545 | 43.831 | **28.403** | 23.876 | 31.016 |

The row-IID column is a labeled negative control, not a certificate.  It is
2.613 minutes tighter solely because it counts repeated cases as independent.
The synthetic dependence stress test demonstrates why that shortcut is
invalid.

The support-free column is a separate five-node family.  It truncates
independent block witness scores rather than outcomes and therefore does not
use the `[0,900]` cap.  It improves the global and family diagnostics and is
essentially tied at procedure level, but loses after note-driven cell
fragmentation.  It cannot be max-selected with the range-based family without
a joint alpha ledger.  All columns remain exploratory on this retrospective
cohort.

As with the unique-patient analysis, decreasing in-sample SAA under a finer
representation is partly optimization bias.  These numbers do not establish
that notes lower population Bayes risk.  The correct magnitude route still
requires an independently evaluated fixed note policy or another theorem that
controls representation selection.

## Metamorphic checks

- Token-order shuffling is exactly invariant for the registered bag profile.
- Appending a target-proportional pure-X token is exactly invariant after the
  canonical parser removes pure-X noise.
- Exposing the X count itself is refused because it is target-derived.

This is the desired behavior: perturbations matter only through the registered
representation, while a representation that consumes the target fails the
outcome-blindness contract.

Aggregate result SHA-256:
`ba744408fbe7a7fba23f1364cd7d61e54161f88d8aec9131c586db6f966dd0fc`

No note text, patient identifier, case identifier, outcome, or row-level
endpoint is stored in the repository artifact.
