#!/usr/bin/env python3
"""Sync the vendored engine into the Pythia plugin and check the release is consistent.

    python3 tools/build_plugin.py            # copy pythia/ -> plugins/pythia/vendor/pythia, run checks
    python3 tools/build_plugin.py --check    # checks only (CI): vendor up to date, version in CHANGELOG
    python3 tools/build_plugin.py --zip DIR  # also write DIR/pythia-<version>.plugin (for Cowork upload)

The plugin is self-contained because marketplace installs copy only plugins/pythia/.
"""
import argparse, filecmp, hashlib, json, re, shutil, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "pythia"
VENDOR = PLUGIN / "vendor" / "pythia"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def tree_hash(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(str(f.relative_to(p)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--zip")
    a = ap.parse_args()
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    version = manifest["version"]
    errors = []
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        errors.append(f"plugin.json version {version!r} is not semver")
    if f"## [{version}]" not in (PLUGIN / "CHANGELOG.md").read_text():
        errors.append(f"CHANGELOG.md has no '## [{version}]' entry")
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    mv = [p.get("version") for p in market["plugins"] if p["name"] == "pythia"]
    if mv and mv[0] != version:
        errors.append(f"marketplace.json lists pythia {mv[0]}, plugin.json says {version}")
    if a.check:
        if not VENDOR.exists() or tree_hash(VENDOR) != tree_hash(ROOT / "pythia"):
            errors.append("vendor/pythia is out of date: run python3 tools/build_plugin.py")
    else:
        if VENDOR.exists():
            shutil.rmtree(VENDOR)
        shutil.copytree(ROOT / "pythia", VENDOR, ignore=IGNORE)
        (PLUGIN / "vendor" / "ENGINE_HASH").write_text(tree_hash(VENDOR) + "\n")
        r = subprocess.run([sys.executable, str(PLUGIN / "scripts" / "pythia_cert.py"), "version"],
                           capture_output=True, text=True)
        if r.returncode or "vendor" not in r.stdout:
            errors.append(f"CLI smoke test failed: {r.stdout}{r.stderr}")
    if errors:
        print("\n".join("ERROR: " + e for e in errors)); sys.exit(1)
    if a.zip:
        out = Path(a.zip) / f"pythia-{version}.plugin"; out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(PLUGIN.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts and f.name != ".DS_Store":
                    z.write(f, f.relative_to(PLUGIN))
        print("wrote", out)
    print(f"pythia plugin {version}: OK")


if __name__ == "__main__":
    main()
