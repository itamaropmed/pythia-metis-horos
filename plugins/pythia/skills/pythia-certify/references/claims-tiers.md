# Claims tiers — what may be said

| Tier | Claim | Needs | Wording |
|---|---|---|---|
| C1 conditional | Region interval and direction for the calibration population of the routed region | A1 entities independent, A2 bounded loss, outcome-blind hierarchy, frozen model | "With 95% simultaneous confidence, the model's miss rate in this region of the calibration population lies in [L, U]" |
| C2 conditional on Λ | Same interval for a later population of that region | C1 + declared odds-shift budget Λ: odds(later) ∈ [odds(L)/Λ, Λ·odds(U)] | "If within-region drift is at most Λ in odds, the later rate lies in …"; always give Λ\* |
| C3 anytime-valid | "This region's later risk exceeds its certified upper bound" | later losses conditionally independent (first case per patient) | "Monitor alarm: evidence of harmful shift (Ville bound, level α)" |
| C4 descriptive | Development-to-later agreement counts, correlations | none | "Descriptively, …" — never used to select |
| — | Individual-case outcome guarantee | not claimed | refuse |
| — | Correlation threshold between risk and error | not claimed | descriptive only |

Categories (local-risk): `certified_higher_than_global` (contrast lower > 0), `certified_lower_than_global`
(contrast upper < 0), `local_shift_unresolved`, `global_only` (case fell to global), `refused` (attestation failed).

Evidence status: on Mayo HRS (retrospective temporal audit, not prospective) the v2 route certified 1,418/2,599
later cases (basic) with later direction agreement 12/12; in Mayo-shaped plasmode the old v1 route produced wrong
directions in 24/25 replicates, v2 in 0/25. Describe Mayo results as a retrospective audit, not confirmation.
