# Maintaining the Pythia plugin

The repository is also a private Claude plugin marketplace (`.claude-plugin/marketplace.json`) that serves
`plugins/pythia`. Teammates install once; every push to `main` with a version bump becomes an update for them.

## Release checklist
1. Change the engine (`pythia/`) and/or the plugin (`plugins/pythia/`). Run `pytest`.
2. Choose the version (semver):
   - PATCH (0.1.1): bug fixes, wording.
   - MINOR (0.2.0): new certificate, option or output field.
   - MAJOR (1.0.0): a claim, default or output field changes meaning (users must re-read old results).
3. Set it in **both** `plugins/pythia/.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json`.
4. Add a `## [x.y.z] - YYYY-MM-DD` entry at the top of `plugins/pythia/CHANGELOG.md` with
   `### New`, `### Changed` (meaning changes — be explicit), `### Fixed`, `### Known limits`.
   This text is what users see in their next session.
5. `python3 tools/build_plugin.py --zip dist` — re-vendors the engine, checks versions/CHANGELOG, smoke-tests the CLI,
   and writes `dist/pythia-x.y.z.plugin` for Cowork users.
6. Commit, tag and push:
   ```
   git add -A && git commit -m "pythia plugin x.y.z: <summary>"
   git tag pythia-vx.y.z && git push && git push --tags
   ```
7. Optional: attach `dist/pythia-x.y.z.plugin` to a GitHub release with the CHANGELOG entry.

## How users get updates
- Claude Code: `/plugin marketplace update pythia-metis-horos` (or marketplace auto-update) then the plugin updates
  to the new version. Claude Code caches by version — **if you don't bump the version, users won't get the change.**
- Cowork: install the new `.plugin` file.
- On their next session the SessionStart hook compares the installed version with the last one they saw
  (stored in the plugin data folder) and injects the new CHANGELOG entries; the `pythia-whats-new` skill answers
  "what's new in Pythia" on demand.

## Rules
- Never commit PHI, derived data, row-level outputs or tokens. `.gitignore` blocks the usual paths; review
  `git status` before every commit.
- `tools/build_plugin.py --check` must pass before pushing (vendor in sync, versions consistent).
