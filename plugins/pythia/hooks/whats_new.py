#!/usr/bin/env python3
"""SessionStart hook: if the installed Pythia plugin version is newer than the last version this user saw,
inject the CHANGELOG entries in between into the session and show a one-line notice. Never fails the session."""
import json, os, re, sys
from pathlib import Path


def vtuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)


def main():
    root = Path(os.environ.get("CLAUDE_PLUGIN_ROOT", Path(__file__).resolve().parent.parent))
    version = json.loads((root / ".claude-plugin" / "plugin.json").read_text())["version"]
    data = Path(os.environ.get("CLAUDE_PLUGIN_DATA") or Path.home() / ".claude" / "plugin-state" / "pythia")
    data.mkdir(parents=True, exist_ok=True)
    marker = data / "last_seen_version"
    seen = marker.read_text().strip() if marker.exists() else None
    if seen == version:
        return
    log = (root / "CHANGELOG.md").read_text()
    parts = re.split(r"(?m)^## \[", log)[1:]
    new = []
    for p in parts:
        v = p.split("]", 1)[0]
        if seen is None or vtuple(v) > vtuple(seen):
            new.append("## [" + p.strip())
    if seen is None:
        new = new[:1]                                   # first install: show only the current release
    marker.write_text(version)
    head = (f"Pythia plugin installed (v{version})." if seen is None
            else f"Pythia plugin updated v{seen} -> v{version}.")
    ctx = head + " Changes since the user last saw it:\n\n" + "\n\n".join(new) + \
        "\n\nMention this update briefly at a natural point if the user works with Pythia."
    print(json.dumps({"systemMessage": head + " Ask \"what's new in Pythia\" for details.",
                      "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
