"""Public aggregate Mayo lock; private recomputation lives in test_guarded.py."""
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "docs/results/mayo_frozen_ci_anchor.json"


def test_frozen_mayo_anchor_is_nonclinical_and_consistent():
    anchor = json.loads(ANCHOR.read_text())
    manifest = json.loads((ROOT / "docs/MAYO_DATA_MANIFEST.json").read_text())
    assert anchor["claim_tier"] == "calibrated"
    assert anchor["expected"] == {
        "best_oof_mae": pytest.approx(32.87001170332562),
        "global_attainable_floor_estimate": pytest.approx(31.118642120070444),
        "stratified_calibrated_bound": pytest.approx(28.84959960434628),
        "stratified_qcap_conditional_bound": pytest.approx(24.186725940850426),
    }
    assert set(anchor) == {"version", "privacy", "claim_tier", "expected", "source_sha256"}
    for name, expected_hash in anchor["source_sha256"].items():
        assert len(expected_hash) == 64
        if name.startswith("data/"):
            assert manifest["files"][name] == expected_hash
            if (ROOT / name).exists():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash
        else:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash
