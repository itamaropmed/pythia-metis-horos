# Cell-local concentration confirmation

Date: 2026-10-14

This is the frozen confirmation registered in
`CELL_RANGE_CONFIRMATION_REGISTRATION.md`.  The analysis used 50 fresh seeds
(`62001`--`62050`) for each of ten known-law mechanisms and the first eight
outcome-blind representation interventions.  The registered cell-local
certificate and the previous global-range certificate were evaluated on the
same 4,000 rows.

## Primary results

| Intervention | Cell-local median bound / gold | Global-range median | Strict paired wins | Positive bounds |
|---|---:|---:|---:|---:|
| baseline | 0.704451 | 0.529814 | 500 / 500 | 500 / 500 |
| bijective cipher | 0.704451 | 0.529814 | 500 / 500 | 500 / 500 |
| canonical nuisance removal | 0.704451 | 0.529814 | 500 / 500 | 500 / 500 |
| naive nuisance retention | 0.120372 | 0.000000 | 417 / 500 | 417 / 500 |
| family garbling | 0.715021 | 0.593375 | 500 / 500 | 500 / 500 |
| within-family reassignment | 0.705290 | 0.581130 | 500 / 500 | 500 / 500 |
| global misalignment | 0.681253 | 0.681253 | 0 / 500 | 500 / 500 |
| bag-token shuffle | 0.704451 | 0.529814 | 500 / 500 | 500 / 500 |

- Known-floor exceedances: **0 / 4,000**.
- Strict improvement over the matched global-range certificate: **3,417 /
  4,000** rows.  Global misalignment is an intentional equality case because
  every label has the full outcome support.
- Baseline mechanism medians improved in **10 / 10** mechanisms.  A two-sided
  sign test at the mechanism level gives `p = 0.001953125`; seeds are not
  treated as independent scientific replicates.
- The previous high-SNR failure was repaired: **50 / 50** baseline bounds were
  positive, versus zero under the global cap.
- Baseline, bijective cipher, canonical nuisance and bag-token representations
  produced exactly equal bounds in **500 / 500** matched cases.

The result is a confirmation of the stated finite-cell theorem under analytic
cellwise hard support ranges.  It is not evidence that such ranges can be
learned from the certification outcomes.  A clinical application must obtain
them from external domain constraints or downgrade/refuse.  The very weak
naive-nuisance result is also informative: mathematically irrelevant random
identifiers fragment the cells and destroy finite-sample power even though the
population information is unchanged.

## Reproducibility

- Registered cell-local raw output:
  `cell_range_confirmation_results.jsonl`
- Matched global-range comparator:
  `global_range_confirmation_comparator.jsonl`
- Cell-local SHA-256:
  `dbb5f4f3162d9d813273d603987d8720f3fec313652e9e7d673a3bfa30b7a20e`
- Comparator SHA-256:
  `43a3d1b24b3d707a299ba29fcc9d829747b501a57926e4214b1d390cc71a412d`

The target-reveal interventions belong to the earlier 5,000-row battery.  All
1,000 literal or unary-encoded target cases were refused rather than allowed to
produce a misleading certificate.
