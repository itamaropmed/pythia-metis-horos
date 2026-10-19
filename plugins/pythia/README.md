# Pythia plugin

Lets Claude (Claude Code and Cowork) run Pythia certificates for you and explain them at the right claim tier.

## What it does
- **pythia-certify**: asks you to confirm the assumptions, runs `local-risk`, `nli` or `table-floor` on a CSV,
  and reports intervals, certified directions (with tipping Λ\*), refusals and what is *not* claimed.
- **pythia-whats-new**: explains what changed in the version you have.
- **Session-start notice**: after an update, Claude sees the new CHANGELOG entries and tells you once.

## Install (team members)
Claude Code:
```
/plugin marketplace add itamaropmed/pythia-metis-horos
/plugin install pythia@pythia-metis-horos
```
The repository is private: you need read access on GitHub and a working `git` login (`gh auth login` or SSH).

Cowork: install the `pythia-<version>.plugin` file from the release (drag it into a chat and accept), or add the
marketplace if your workspace supports GitHub marketplaces.

## Update
```
/plugin marketplace update pythia-metis-horos
/plugin update pythia@pythia-metis-horos
```
(or enable auto-update for the marketplace in `/plugin` → Marketplaces). The next session shows what's new.

## Requirements
Python 3.10+, `numpy`, `scipy`. The Pythia engine is vendored in `vendor/` — no other install needed.

## Privacy
Aggregates only, ≥ 50 entities per reported region/cell, no identifiers or note text in outputs.
Run it on the machine that holds the data.
