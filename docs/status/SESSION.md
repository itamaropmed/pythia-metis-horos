# Provenance

This project was scaffolded and M0 was written in a Cowork session with Claude.

**Conversation:** https://claude.ai/code/session_01DPT8cwk1Drvpwz4NeHLGp2

That thread holds the reasoning behind decisions that are only stated as
conclusions in the code and docs:

| topic | where it matters |
|---|---|
| why the loss router branches Bregman / guarded / refuse | `pythia/core/router.py` |
| the schedule-derivation ablation and leakage audit of the booked slot | `docs/status/PLAN.md`, client track C1 |
| why arm B (tabular + 11 note features) beats arm C (+ note-text SVD) | `config.yaml`, `docs/experiments/RESULTS2.md` |
| the A-fold failure on interaction designs, and the three rejected Δ estimators | `docs/experiments/M0_FINDINGS.md` |
| literature map: conformal risk control, KNIFE-P/LMC-P, CAAFE, Domino, MLE-STAR | `docs/status/PLAN.md` validation + M5 |
| the incumbent Mayo duration model spec (held-out MAE 31.2623) | `docs/experiments/RESULTS2.md` context |

Date scaffolded: 2026-09-10.
