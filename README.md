# Pythia / Metis / Horos

> **Naming (2026-10-18).** The layers were renamed after figures of Greek myth. **Pythia** (formerly *Peras*) is the Oracle of Delphi: it answers *how well can any model possibly do here?* before you train one. **Metis** (formerly *Poros*) is the Titaness of cunning counsel and, in Plato's *Symposium*, the mother of Poros: it plans the search for tighter certificates. **Horos** is unchanged. The old import names `peras` and `poros` still work as deprecated aliases. Research folders (`research/peras_v5/`, …), frozen registrations and result logs keep their historical names.

**Certified lower bounds and attainability estimates for supervised learning under a
user-chosen loss.**

Given a dataset, a loss, and a frozen representation, this system answers a question
that model selection cannot: *how well can any model possibly do here, and how close
is the one you have?* It returns a certified floor `B`, an estimate `Ê` of the
attainable floor, and a labelled claim tier — or it refuses, and says why.

Maintainer: **Itamar Zernitsky** — Department of Mathematics, Bar-Ilan University;
Opmed.ai (Shadows research team).
<itamar.zernitsky@opmed.ai> · +972 54-797-1303 · [@itamaropmed](https://github.com/itamaropmed)

---

## Three layers, deliberately separable

| layer | what it is | status |
|---|---|---|
| **pythia** | the theory and pure computation of `B`, `Ê`, sensitivity, and refusals. Never trains a model. | implemented |
| **metis** | evidence-gated search plus frozen per-case model-risk routing | first risk-map POC implemented; autonomous search still deferred |
| **horos** | the user-facing application: ingest → certificate → slice exploration | implemented |

### What the system promises

1. **B** — a tiered lower bound. A result is called *certified* only when its theorem
   and stated conditions support that word.
2. **Ê** — an estimate of the attainable floor: a well-built model should reach about `Ê`.
3. **Γ = Ê − B** — the remaining uncertainty, and the agent's entire objective.

Every certificate is labelled `proved`, `conditional`, `calibrated`, `exploratory`,
or `refused`. The refusal path is a first-class outcome: losses and data regimes where
the method does not apply return *"not certifiable, here is why, here is what to use
instead"* rather than a number.

---

## Claude plugin and team marketplace

This repository is also a private Claude plugin marketplace. The `pythia` plugin (`plugins/pythia/`) lets Claude
run Pythia certificates (`local-risk`, `nli`, `table-floor`) with mandatory attestations, claim tiers and
aggregate-only privacy rules, and tells users what changed after every update.

```
/plugin marketplace add itamaropmed/pythia-metis-horos
/plugin install pythia@pythia-metis-horos
```

Release process: `MAINTAINING.md`. User-facing changes: `plugins/pythia/CHANGELOG.md`.

## Repository layout

```
pythia/        core     loss router, Bayes risk, envelopes, rungs, stratification
              suite    certification suite (seeded generators + versioned registry)
              compare  literature comparators
              guards   audited constants and guard artifacts
metis/        agent    loop, planner, admissible action library a1..a7
              gate     validity / non-inferiority / provenance predicates G1..G5
              memory   case base, bi-encoder recall, cross-encoder rerank
horos/        ingest   loaders, task card, EDA adapter, leakage screen
              represent tabular lane, NL lane, fusion
              models   model zoo; all fitting and cross-fitting lives here
              slice    partitions, clusters, kNN communities, discovery
              report   certificate, declaration, visualisation
              benchmark synthetic design families with analytic Bayes gold
papers/       pdf/     compiled reader PDFs
              tex/     manuscript sources
docs/         theory/       derivations and bound development
              audits/       line-by-line theorem and scope audits
              experiments/  frozen result reports
              results/      machine-readable result JSON
              status/       plan, implementation status, session log
scripts/      reproduction and comparison drivers
tests/        53 test modules, including a static architecture guard
notebooks/    exploratory notebooks
```

No patient data and no derived artifacts are tracked in Git. `.gitignore` excludes
`data/raw/`, `data/derived/`, `*.parquet`, `*.npy`, `*.pkl`. See
`docs/MAYO_DATA_MANIFEST.json` and `data/raw/mayo/SOURCES.md` (local) for the data
locations; the 3.7 GB notes parquet lives in S3, not here.

---

## Papers

| document | file |
|---|---|
| **Pythia — decision-risk floors** (ICML-shaped paper + reproducibility supplement, single reader copy) | `papers/pdf/PYTHIA_ICML_COMPLETE_WITH_SUPPLEMENT.pdf` |
| **Pythia/Metis/Horos system blueprint** (workflow, evidence, gates, roadmap, prospective study) | `papers/pdf/PYTHIA_METIS_SYSTEM_BLUEPRINT.pdf` |
| **Metis–Horos formal blueprint** (16 pp; axioms, separation-soundness, α-ledger FWER, claim lattice, gate algebra, no-invalidation and fail-closed theorems, evidence-graph staleness, knapsack allocation, threat model, open problems) | `papers/pdf/METIS_HOROS_FORMAL_BLUEPRINT.pdf` |
| *Where prediction ends* (narrative exposition) | `papers/pdf/where_prediction_ends.pdf` |
| earlier working manuscript | `papers/pdf/pythia_metis_working_paper_2026.pdf` |

Sources for all of these are in `papers/tex/`. The formal blueprint is the deepest
treatment of the Metis/Horos control layer: it states the system invariant as an axiom,
proves that measurability separation is sound, that the α-ledger preserves family-wise
error under agent-chosen stopping, and that no admissible action can invalidate a
standing certificate.

`docs/audits/ICML_JUDGE_RESPONSE_20260914.md` records the proof corrections and the
remaining review risks. These are **not** official ICML-template or anonymised
submission files, and Mayo multi-loss figures remain exploratory.

---

## Headline results

**Ten-design out-of-distribution MAE stress test** (seed 20260928, 20 draws × 10 design
families, n = 1600; analytic Bayes gold evaluated at the sampled covariates):

| method | below known risk | median usable tightness | paired wins |
|---|---:|---:|---:|
| clipped paired 95% | 200/200 | 0.5634 | 0 |
| **clipped grouped 95%** | **200/200** | **0.6405** | **180** |

Twenty draws tied at zero on sparse-20D; grouped won every draw in each of the other
nine families. Replicated on independent seed 20260925 (90 wins / 0 losses / 10 ties).
The 200/200 coverage has a 95% Wilson lower bound of 0.9812. This is a within-benchmark
result against the implemented paired Hoeffding/Bernstein baseline — *not* a claim to
beat all published bounds.

**Multi-loss validity**: the train-free pair/group envelope covers MAE, pinball
(q10/q90), and within-window miss (W15/W30/W60), with 1,000/1,000 valid bounds across
five losses under Bonferroni adjustment over methods, clips, families, and losses.

**Mayo baseline**, Arm B (tabular + 11 note features), n = 12,993 cases / 10,532 patients:

| quantity | point | interval |
|---|---|---|
| `B*` calibrated floor (2-D guard, stratified) | 28.85 | ≥ 28.27 (5%) |
| `Q_cap` guard rung (guard proved; full bound conditional) | 24.18 | ≥ 23.60 (5%) |
| `Ê₂` estimated attainable floor | 31.12 | [30.43, 31.74] |
| best out-of-fold MAE | 32.87 | [32.26, 33.50] |
| temporal holdout MAE | 31.67 | [30.36, 32.85] |

Zero-information baseline MAD(Y) = 80.04 min. W30 attainable-accuracy ceiling 79.81%
against 62.60% achieved.

Full reports: `docs/experiments/`.

---

## Install and run in a clean environment

Requires Python ≥ 3.10.

```bash
git clone https://github.com/itamaropmed/peras-poros-horos.git
cd peras-poros-horos

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"            # add ,mayo for the xgboost==3.3.0 Mayo arm
```

Run the test suite (no data required — the synthetic benchmark is self-contained):

```bash
pytest -q
```

Reproduce the headline synthetic comparison:

```bash
python scripts/run_ood_mae_comparison.py         # ten-design MAE stress test
python scripts/run_decision_loss_comparison.py   # five-loss validity sweep
python scripts/run_m0_suite.py                   # Bregman core acceptance
```

### Mayo arm (requires the restricted data)

```bash
python scripts/check_data.py                     # verify the working set is present
pythia certify config.yaml --arm B
pythia mayo-ladder config.yaml
pythia prepare mayo.study.yaml
python scripts/reproduce_mayo_structured.py CASES.parquet FOLDS.parquet OOF.parquet
```

`pythia prepare` is dataset-agnostic: the study YAML assigns row, target, entity, and
time roles. Mayo-specific behaviour lives in adapters and validation fixtures, not in
the certification core.

---

## The non-training boundary

Pythia accepts outcomes, aligned frozen out-of-fold predictions, fold/entity identifiers,
representation provenance, and optional guard artifacts. It does **not** fit, tune,
embed, prompt, or engineer a predictor. Those operations belong to Horos or to an
external benchmark. `tests/test_architecture.py` statically rejects estimator training
calls and trainer imports anywhere under `pythia/`.

## Scope and honesty policy

- No universal best-bound or best-estimator claim is made.
- Metis autonomy is **designed, not deployed**; the repository ships the gate and action
  scaffolding, not a running agent.
- Mayo results are retrospective diagnostics. No clinical certification is claimed.
- Where a method failed — the sparse-20D tie, the weakly identified (Δ, λ) solver, the
  partial M0 acceptance on interaction designs — the failure is documented in
  `docs/experiments/` rather than removed.

---

## Pythia v5 (October 2026): certified value of clinical text beyond the table

Pythia v5 extends the floor certificates to records with a **table row plus free text**. Start with **[`docs/PYTHIA_V5_GUIDE.md`](docs/PYTHIA_V5_GUIDE.md)**.

| What | Where |
|---|---|
| Guide: concept → theory → code → tests → experiments, plus quick start | `docs/PYTHIA_V5_GUIDE.md` |
| Theory index and dependency graph | `docs/theory/README.md` |
| Condensed statements of all v5 results | `docs/theory/PYTHIA_V5_COMPENDIUM.md` |
| **Natural Language Information (NLI)**: definition, axioms, theorems | `docs/theory/NATURAL_LANGUAGE_INFORMATION_NLI_20261017.md` |
| **PYTHIA-U**: one simultaneous table + text certificate | `docs/theory/PYTHIA_U_UNIFIED_CERTIFICATE_20261018.md`, `pythia/core/unified.py` |
| Roadmap and open audit items | `docs/status/ROADMAP_V5_20261018.md` |
| Discussion and decision log | `docs/status/DISCUSSION_LOG_V5_202610.md` |
| Experiment index and reproduction | `research/peras_v5/README.md` |
| Literature survey (2022–2026) | `docs/literature/Certified risk bounds tabular text.md` |
