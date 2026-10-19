"""Smoke tests for the Pythia plugin CLI (synthetic data only)."""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np

CLI = Path(__file__).resolve().parents[1] / "plugins" / "pythia" / "scripts" / "pythia_cert.py"
ATTEST = ["--attest-outcome-blind-levels", "--attest-independent-entities", "--attest-loss-cap", "--attest-frozen-model"]


def _cal(path, n=4000, seed=0):
    r = np.random.default_rng(seed); pid = r.integers(0, 2500, n); fam = r.choice(list("AB"), n); pb = r.integers(0, 3, n)
    loss = (r.random(n) < .2 + .25 * (fam == "A")).astype(int)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pid", "l1", "l2", "loss"])
        for i in range(n):
            w.writerow([pid[i], f"{fam[i]}|{pb[i]}", fam[i], loss[i]])


def run(*a):
    return subprocess.run([sys.executable, str(CLI), *a], capture_output=True, text=True)


def test_local_risk_certifies_planted_direction(tmp_path):
    _cal(tmp_path / "c.csv")
    r = run("local-risk", "--cal", str(tmp_path / "c.csv"), "--entity-col", "pid", "--levels", "l1,l2",
            "--loss-cols", "loss", "--min-entities", "400", "--out", str(tmp_path / "o"), *ATTEST)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "o" / "pythia_result.json").read_text())
    rel = {x["region"]: x.get("relation") for x in res["regions"]}
    assert any("|A|" in k and v == "certified_higher_than_global" for k, v in rel.items())
    assert not any("|A|" in k and v == "certified_lower_than_global" for k, v in rel.items())
    assert "pid" not in (tmp_path / "o" / "pythia_result.json").read_text()


def test_missing_attestation_refuses(tmp_path):
    _cal(tmp_path / "c.csv")
    r = run("local-risk", "--cal", str(tmp_path / "c.csv"), "--entity-col", "pid", "--levels", "l1,l2",
            "--loss-cols", "loss", "--out", str(tmp_path / "o"))
    assert r.returncode == 2
    assert json.loads((tmp_path / "o" / "pythia_result.json").read_text())["status"] == "refused"
