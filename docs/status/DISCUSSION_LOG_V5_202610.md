# Discussion log: Pythia v5, October 2026 (Cowork sessions with Claude)

This is a chronological record of what was asked, what was decided, what was built, and what was found. User requests are paraphrased; quotes are marked. Results are summarised here; the detail is in the linked documents.

## Standing constraints (set by the user, binding for all work)

- Never commit PHI. Mayo analyses export aggregates only. Real clinical notes never leave the device.
- The folder `PycharmProjects/Clinical_notes` holds private clinical data and is not touched by cloud work. Local, aggregate-only runs are allowed only on explicit request.
- GitHub tokens are one-shot and are never persisted.
- Do not touch Notion.
- Commit or push only when asked. As of 2026-10-18, v5 work is in the working tree on branch `codex/pythia-support-free-cluster` and is **uncommitted**.
- Data preference: free real data (MTSamples, PMLB) and honest comparisons against accepted literature methods.

## Timeline

| # | Request (paraphrased) | Outcome | Documents |
|---|---|---|---|
| 1 | Continue the Codex handover; close the support/concentration gap | Fixed-action domination, then dependency-graph domination (realized-period estimand), then smooth domination for continuous X | `theory/PYTHIA_GAP_PLAN_20261014.md`, `DOMINATED_SAA_20261014.md`, `DEPENDENCY_GRAPH_DOMINATION_20261014.md`, `SMOOTH_DOMINATION_20261015.md`; `research/peras_v5/domination_20261014/`, `graph_20261014/`, `continuous_20261015/` |
| 2 | "Run tests on tabular data and complex natural-language text … find free tabular data and clinical notes" | MTSamples plus PMLB. Generic text metrics give no near-replicates. Exact duplication and templating are a usable replicate resource. Replicate pair certificates | `theory/REPLICATE_PAIR_CERTIFICATES_20261015.md`; `research/peras_v5/realdata_20261015/` |
| 3 | "What worked and in what direction do I need to develop?" | Direction set: modulus-free bounds, text geometry, information relative to the table | `theory/PYTHIA_DIRECTIONS_GEOMETRY_20261015.md` |
| 4 | Tests with known noise and signal; develop the bound into real literature gaps | Equal-floor identity and **PYTHIA-X** (modulus-free extrapolation under A(γ)). It beat the accepted near-replicate route run with the *oracle* modulus in most designs | `theory/MODULUS_FREE_EXTRAPOLATION_20261016.md`; `novelty_20261016/EXTRAPOLATION_RESULTS_20261016.md` |
| 5 | Strengthen the four weak points: adversarial cases, γ with boundaries, slack, squared loss | Prop. 1: no assumption-free fix without replicates, plus a retest anchor. Prop. 2: γ = ½ is safe. Betting bounds for squared loss (0.76–0.86 of gold) | `theory/PYTHIA_X_STRENGTHENING_20261016.md`; `novelty_20261016/STRENGTHENING_RESULTS_20261016.md` |
| 6 | Continue the novelty development of the bound | Model-free sandwich: U1 ceiling and I1 information. MTSamples specialty floor ≤ 0.262, beating every fitted model's certified upper bound | `theory/MODEL_FREE_SANDWICH_20261017.md`; `sandwich_20261017/` |
| 7 | Is there prior work on information and signal in text? Define something new given the tabular signal | Survey (V-information, PID, CODEC and others). Table-anchored text information | `theory/TABLE_ANCHORED_TEXT_INFORMATION_20261017.md`; `anchored_20261017/` |
| 8 | A new NLP theory of information in text conditioned on the table, with a new task; test on 10 OOD generators and real data | **AFI** (Agreement-Field Information). VTC task, extraction-completeness certification. On MTSamples, ≥ 0.075 residual text information beyond four extracted fields | `theory/AGREEMENT_FIELD_INFORMATION_20261017.md`; `afi_20261017/` |
| 9 | MIMIC access? Define the theorem well; dynamic geometric embedding | No MIMIC access from the sandbox (PhysioNet blocked, credentialed). AFI Theorems 1–5 and Props 6–8, including predictable geometry. e5-small-v2 embeddings tested. Local aggregate-only MIMIC script prepared | `theory/AFI_TEXT_THEOREMS_20261017.md`; `dynamic_20261017/`; `scripts/run_vtc_mimic.py` |
| 10 | Bound the three classical ML gaps; per-sample risk, categorical and numeric; a 2022–2026 survey | **PYTHIA-G3** three-gap certificate for any loss, per-sample labels, certified-easy groups. Literature survey written. 0 violations in 105 bundles | `research/peras_v5/g3_20261017/G3_RESULTS_20261017.md`; `docs/literature/Certified risk bounds tabular text.md` |
| 11 | LP bounds for prediction and estimation; relax slowly to 0; information geometry of text given the table | LP-1 (relaxation dial, tipping slack ε\*), LP-2 (moment LP), conditional text map. On MTSamples, text resolves 69% of outcome disagreement in operative notes | `theory/LP_RELAXATION_AND_TEXT_GEOMETRY_20261017.md`; `lp_20261017/` |
| 12 | "I need a new definition of natural language information – NLI … expectation of new introduced information … full development … test its power over unseen OOD datasets versus accepted literature methods" | **NLI** theory: definition, operational theorem, axioms, uniqueness, variational form, certificate, exact test, lower-only theorem. Detection ties the best valid tests (local permutation, WGCM) and beats bootstrap ΔBrier (p ≈ 1e-31) and CODEC. The unique capability is the **certified magnitude**: positive in 61% of unseen signal datasets, against 0.7% for the pair route, with 0 of 600 violations | `theory/NATURAL_LANGUAGE_INFORMATION_NLI_20261017.md`; `nli_20261017/NLI_RESULTS_20261017.md` |
| 13 | Combine that capability with the whole Pythia system, conditional on the table signal bound, into one valid tight bound | **PYTHIA-U** unified LP. 0 of 50 violations. Certified NLI is 42% of truth, against 5–11% before; ceiling 1.05–1.14 × truth. On real data with a rich table, text information doubles (0.21 vs 0.10–0.14) | `theory/PYTHIA_U_UNIFIED_CERTIFICATE_20261018.md`; `unified_20261018/UNIFIED_RESULTS_20261018.md` |
| 14 | Save everything to the PyCharm project with organised theory, planning and discussion documents, like a library | This log, `docs/PYTHIA_V5_GUIDE.md`, `docs/theory/README.md`, `docs/theory/PYTHIA_V5_COMPENDIUM.md`, `docs/status/ROADMAP_V5_20261018.md`, `research/peras_v5/README.md` | — |
| 15 | Rename Peras and Poros after figures of Greek mythology | **Peras → Pythia** (the Delphic oracle: how well can any model possibly do?) and **Poros → Metis** (cunning counsel; Poros's mother in Plato's *Symposium*); Horos unchanged. Full code rename with deprecated `peras`/`poros` import aliases; the Mayo CI anchor was re-hashed after verifying the change is a pure rename (`docs/results/RENAME_PROVENANCE_20261018.json`); research folder names and frozen registries kept | `README.md` naming note |

## Errors found and fixed along the way (for reproducibility)

| Error | Fix |
|---|---|
| Level t ≥ 1 pairs in `certificate_v2` gave spurious bounds | Skipped |
| Row order leaked the label: the MTSamples CSV is sorted by specialty | Row shuffle, plus `perm_seed` in the nested matchings |
| Greedy matchers paired a row with itself when points were tied | Self-pairs excluded and the matching asserted perfect. The concept-dictionary bound was corrected and an earlier claim that A(½) had been refuted was withdrawn |
| An NLI cohort bound with a constant score gave 0/0 round-off (strata with a single class) | Threshold on H |
| BLAS thread oversubscription | `OMP_NUM_THREADS=1` |

## Honest negative findings to remember

- The joint *floor* from text needs an assumption. This is a theorem (AFI Thm 4 / NLI Thm 7); at γ = ½ it is weak on real notes.
- NLI does not beat the best valid *tests* at detection. Its win is the guaranteed magnitude.
- Neural sentence embeddings (e5-small) carried less outcome information than TF-IDF on MTSamples specialty.
- Stratified (table-conditional) certificates are valid but looser globally. Use them as maps.
- An open audit item on U-statistic concentration for non-identical rows: see `theory/PYTHIA_V5_COMPENDIUM.md` §F.
