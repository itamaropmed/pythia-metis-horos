# Changelog — Pythia plugin

All notable changes for users of the plugin. Newest first. Versions follow semver:
MAJOR = a claim, default or output field changed meaning; MINOR = new certificate or option; PATCH = fixes.

## [0.1.0] - 2026-10-19

### New
- `pythia-certify` skill: runs three certificates on a local CSV and writes aggregate JSON + a markdown report.
  - **local-risk** — v2 route: outcome-blind support partition, eligible-node α ledger, direct local-vs-global
    contrast, odds transport Λ and tipping Λ* per certified direction.
  - **nli** — NLI cohort lower bound (text information beyond the table) with the table-floor box and a held-out
    model ceiling, giving a certified upper bound on the joint (table + text) floor.
  - **table-floor** — table and marginal Brier floors and table information.
- Mandatory attestations (`--attest-*`): a missing one yields `refused`, never a silent weaker answer.
- Privacy defaults: aggregates only, no identifiers, no region/cell below 50 entities; row-level cards only on request.
- `pythia-whats-new` skill and a session-start notice when the plugin version changes.

### Known limits
- Λ transport and tipping Λ* are defined for 0/1 losses only (loss cap 1).
- NLI is lower-only; a zero bound is not evidence that text carries no information.
