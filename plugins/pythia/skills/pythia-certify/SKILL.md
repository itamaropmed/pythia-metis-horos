---
name: pythia-certify
description: This skill should be used when the user asks to "run a Pythia certificate", "certify local model risk", "which subgroups is my model worse in", "W30 risk by region", "how much do the notes/text add beyond the table", "NLI bound", "table floor", "Brier floor", or wants a statistically valid (not just estimated) statement about a frozen model's error, a subgroup's risk, or text information beyond tabular data.
---

# Pythia certificates

Pythia turns a calibration file into simultaneous, assumption-graded bounds. It never trains models. Run it
through the bundled CLI and report its claims at the tier it states — never stronger.

## 1. Pick the certificate

| User wants | Subcommand | Input CSV (one row per case) |
|---|---|---|
| Risk intervals per subgroup/region for a frozen model; "where is it worse than overall" | `local-risk` | entity id, hierarchy columns (finest first), one loss column per model (e.g. 1 if abs error > 30 min) |
| How much text/notes add beyond the table; joint table+text floor | `nli` | table cell, class label, held-out class probabilities `p_<k>` from a text-aware model |
| Best achievable Brier from the table alone | `table-floor` | table cell, class label |

Read `references/input-contract.md` before preparing data. Read `references/local-risk-v2.md` or
`references/nli-and-floors.md` for the theorem behind the chosen subcommand when the user asks why.

## 2. Check the attestations with the user

Each subcommand requires explicit `--attest-*` flags. Do not add a flag unless the user has confirmed the fact
(or it is evident from code you can read). Ask in plain language, e.g. "Was the model fitted on patients that are
not in this calibration file?" Missing attestations make the CLI return `refused` — report that, explain what
is missing, and stop. Never work around a refusal.

| Flag | Meaning |
|---|---|
| `--attest-outcome-blind-levels` | hierarchy/bins built without outcomes (from predictions or covariates) |
| `--attest-independent-entities` / `--attest-independent-rows` | different patients independent; for `nli`/`table-floor` one row per patient |
| `--attest-loss-cap` | every loss in [0, `--loss-cap`] |
| `--attest-frozen-model` | certified model = deployed model, fitted on entities disjoint from CAL |
| `--attest-heldout-probs` | probabilities produced without these rows' labels |

Fix the support threshold `--min-entities` (default 100) and α **before** looking at results. Do not rerun with
other thresholds and keep the best; if the user wants sensitivity, declare the list up front and report all.

## 3. Run

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/pythia_cert.py" local-risk --cal cal.csv --entity-col patient \
  --levels family_pred_dis,family_pred,pred_bin --loss-cols loss_basic,loss_llm --out pythia_out \
  --attest-outcome-blind-levels --attest-independent-entities --attest-loss-cap --attest-frozen-model
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/pythia_cert.py" nli --data rows.csv --cell-col cell --y-col y \
  --prob-prefix p_ --attest-heldout-probs --attest-independent-rows --out pythia_out
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/pythia_cert.py" table-floor --data rows.csv --cell-col cell --y-col y \
  --attest-independent-rows --out pythia_out
```

Add `--route new_cases.csv` to `local-risk` to route new cases into frozen regions (counts only). Requires
Python 3.10+ with numpy and scipy (`pip install numpy scipy`). If the data lives on the user's computer and is
sensitive, run the CLI there, not in a cloud workspace.

## 4. Report

Read `pythia_out/pythia_report.md` and summarise. Follow `references/claims-tiers.md` exactly:

- State the claim tier the result carries (C1 conditional, C2 conditional on Λ, C3 monitor, C4 descriptive).
- For each certified direction, give Λ\* ("survives up to Λ\* odds drift within the region").
- `local_shift_unresolved` means "not resolved at this sample size", never "same as overall".
- An NLI lower bound of 0 means "not certified", never "text adds nothing".
- Never present an interval as a guarantee for an individual patient.
- Never claim a correlation threshold between risk and error.

## 5. Privacy (always)

Follow `references/privacy.md`: aggregates only; no names, IDs, dates or note excerpts in chat or files you
share; no region or cell below 50 entities; row cards (`--row-cards`) stay on the user's machine. If input looks
like it contains direct identifiers in non-ID columns, stop and ask.
