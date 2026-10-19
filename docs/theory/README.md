# Theory index

**Status tags**
- **core**: in use by the current certificate.
- **superseded**: kept for history; a later note replaces it.
- **background**: earlier Pythia foundations.

For condensed statements of every v5 result, see `PYTHIA_V5_COMPENDIUM.md`. The guide is `../PYTHIA_V5_GUIDE.md`.

## Dependency graph (v5)

```
REPLICATE_PAIR_CERTIFICATES ─┐
                             ├─> MODULUS_FREE_EXTRAPOLATION (PYTHIA-X, equal-floor identity, A(γ))
                             │        └─> PYTHIA_X_STRENGTHENING (γ = ½ safety, betting bounds, retest anchor)
                             │                 └─> MODEL_FREE_SANDWICH (U1 ceiling, I1 information)
                             │                          └─> TABLE_ANCHORED_TEXT_INFORMATION
                             │                                   └─> AGREEMENT_FIELD_INFORMATION ─> AFI_TEXT_THEOREMS
                             │                                            (no-free-lunch, dynamic geometry, posterior map)
                             │                                                     ├─> LP_RELAXATION_AND_TEXT_GEOMETRY (LP-1/LP-2, ε dial)
                             │                                                     └─> NATURAL_LANGUAGE_INFORMATION_NLI
                             │                                                               └─> PYTHIA_U_UNIFIED_CERTIFICATE
DOMINATED_SAA ─> DEPENDENCY_GRAPH_DOMINATION ─> SMOOTH_DOMINATION   (table-side routes for dependent / continuous data)
```

## v5 notes, chronological

| Date | Note | One-line content | Status |
|---|---|---|---|
| 2026-10-14 | `PYTHIA_GAP_PLAN_20261014.md` | Survey-based plan to close the nonlinear-SAA concentration gap | background |
| 2026-10-14 | `DOMINATED_SAA_20261014.md` | Fixed-action domination: range inequalities applied one-sidedly to an additive sum that dominates the SAA optimum | core (table floor for any loss) |
| 2026-10-14 | `DEPENDENCY_GRAPH_DOMINATION_20261014.md` | Realized-period estimand, dependency-graph concentration; removes iid and stationarity | core (dependent data) |
| 2026-10-14 | `CLUSTER_CONCENTRATION_GAP_20261014.md`, `FINITE_REPRESENTATION_LATTICE_20261014.md`, `REPRESENTATION_INTERVENTION_PROGRAM_20261014.md` | Gap statement, finite lattice and refusals, intervention program | background |
| 2026-10-15 | `SMOOTH_DOMINATION_20261015.md` | Second-order drift penalties for continuous information | core (continuous X) |
| 2026-10-15 | `REPLICATE_PAIR_CERTIFICATES_20261015.md` | 0-1 floors from exact replicates, nuisance quotients, unbiased minorants | core |
| 2026-10-15 | `PYTHIA_DIRECTIONS_GEOMETRY_20261015.md` | Catalog of geometry, topology and information directions; rule that outcome-blind geometry only changes tightness | background |
| 2026-10-16 | `MODULUS_FREE_EXTRAPOLATION_20261016.md` | **Equal-floor identity** E Q(M) = Ḡ + D(M); PYTHIA-X modulus-free extrapolation under A(γ) | core |
| 2026-10-16 | `PYTHIA_X_STRENGTHENING_20261016.md` | Variogram/nugget reading; Prop. 1 (no assumption-free fix without replicates); Prop. 2 (γ = ½ safe for smooth-plus-jump laws); betting bounds | core |
| 2026-10-17 | `MODEL_FREE_SANDWICH_20261017.md` | Training-free ceiling U1 and information lower bound I1 | core |
| 2026-10-17 | `TABLE_ANCHORED_TEXT_INFORMATION_20261017.md` | First definition of "signal in text" measured against the table | superseded by AFI and NLI |
| 2026-10-17 | `AGREEMENT_FIELD_INFORMATION_20261017.md` | AFI: agreement kernel, pair designs, ΔI(T\|X), asymmetry principle, resolution spectrum, co-information, VTC task | core |
| 2026-10-17 | `AFI_TEXT_THEOREMS_20261017.md` | Theorems 1–5 and Props 6–8: no-free-lunch for text, predictable geometry valid (martingale), posterior map optimal | core |
| 2026-10-17 | `LP_RELAXATION_AND_TEXT_GEOMETRY_20261017.md` | LP-1 design LP with ε relaxation and tipping slack; LP-2 moment LP; conditional fibre geometry | core |
| 2026-10-17 | `NATURAL_LANGUAGE_INFORMATION_NLI_20261017.md` | **NLI** definition, operational theorem, axioms A1–A7, uniqueness of the quadratic NLI, variational form, finite-sample certificate, exact test, lower-only theorem | core |
| 2026-10-18 | `PYTHIA_U_UNIFIED_CERTIFICATE_20261018.md` | **PYTHIA-U**: one simultaneous LP over table, pair, NLI, model and A(γ); falsification; table-conditional form | core |

## Earlier Pythia foundations (pre-v5)

These notes are background:
- `REPLICATE_BOUND.md`
- `DEPENDENCE_BOUND.md`
- `DEPENDENCE_PROFILE.md`
- `CURVATURE_JENSEN_CERTIFICATE.md`
- `THEORY_AND_SYNTHETIC_PROGRAM.md`
- `THEORY_GAP_CLOSURE_20260913.md`

The Pythia v5 summary documents are `research/peras_v5/THEORY_v5.md`, `RESULTS_v5.md` and `RESULTS_v5b.md`.
