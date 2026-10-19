---
name: pythia-whats-new
description: This skill should be used when the user asks "what's new in Pythia", "what changed in the Pythia plugin", "Pythia release notes", "which version of Pythia do I have", or after a session-start notice that the Pythia plugin was updated.
---

# What's new in Pythia

1. Read `${CLAUDE_PLUGIN_ROOT}/.claude-plugin/plugin.json` for the installed version.
2. Read `${CLAUDE_PLUGIN_ROOT}/CHANGELOG.md`.
3. Summarise the installed version's entry first, in plain language: what is new, what changed meaning
   (anything under "Changed" or a MAJOR bump — call these out explicitly, since earlier results may need
   re-reading), and known limits. Offer older entries only if asked.
4. If the user wants the newest version, tell them to update the plugin from its marketplace
   (Claude Code: `/plugin marketplace update pythia-metis-horos` then `/plugin update pythia@pythia-metis-horos`;
   Cowork: reinstall or update from the plugin panel). Never claim a newer version exists without seeing it.
