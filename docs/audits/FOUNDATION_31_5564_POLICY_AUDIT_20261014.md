# Mayo 31.5564 held-out policy audit

Date: 2026-10-14

The saved verifier for the clinical-notes project passed without modification:

```text
heldout_rows=2839
heldout_mae=31.556418073067334
status=PASS
```

The selected policy was chosen by training OOF performance and evaluated on a
later temporal cohort whose patients do not overlap training.  Its metadata
records 10,882 training cases, 2,839 held-out cases, zero patient overlap, the
strict rule `note_date < scheduled_in_room - 2 days`, no postoperative notes,
and no old 11-variable LLM feature table.

This resolves the existence and numerical reproducibility of **31.5564**.  It
does not validate the separate historical 31.2623 claim.  The held-out cohort
was viewed during multiple retrospective iterations, so 31.5564 is a strong
retrospective estimate rather than independent prospective confirmation.

## Fixed-policy information-value diagnostic

The new Pythia bridge was run with the saved policy losses and an explicit
exploratory sampling override.  Pythia did not train or alter the policy.

| Coarse information | Cells | Coarse SAA | Coarse lower | Policy upper | Value lower |
|---|---:|---:|---:|---:|---:|
| procedure family | 10 | 53.1663 | 30.2263 | 33.8533 | **0.0000** |
| procedure | 112 | 40.6252 | 17.6853 | 33.8533 | **0.0000** |

The policy MAE is 31.5564 and its dKelly upper endpoint under the working
`[0,900]` cap is 33.8533.  Neither coarse lower endpoint exceeds that upper
endpoint, so Pythia correctly refuses to claim a positive information-value
magnitude.

This zero is a resolution result, not evidence that structured data and notes
have no value.  Procedure-cell fragmentation and the global 900-minute support
penalty are especially costly.  A positive clinical certificate requires
externally defensible cell-local supports and a new evaluation cohort that was
not used during development.

## Hashes

- Result metadata:
  `1bebddb50be8657eb7a8722a5e58367cffaa29a440ea4c54e7c95f1ec57b66b3`
- Saved held-out predictions:
  `330fba1969af2923c601a219d91ea46b3ac183365e1c0f31f143d6bea38e2df7`
- Aggregate Pythia diagnostic:
  `8b29c9ef9e78bbd02274e967e0694df1f5765908988a73253d569de774eadc41`

No row-level prediction file was copied into the Pythia repository.
