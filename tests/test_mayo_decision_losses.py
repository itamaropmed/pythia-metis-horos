from pathlib import Path

import pytest

from horos.benchmark.mayo_decision_losses import run_mayo_decision_losses


DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived" / "mayo"


@pytest.mark.skipif(not (DERIVED / "PB.npy").exists(), reason="private Mayo artifacts unavailable")
def test_retrospective_mayo_grouped_loss_audit_reproduces_aggregate_anchor():
    result = run_mayo_decision_losses(DERIVED)
    assert result["claim_tier"] == "exploratory"
    assert result["n_unique_patients"] == 10532
    assert len(result["periods"]) == 4
    first = result["periods"][0]["representations"]["tabular_metric"]["losses"]
    assert first["w30_miss"]["grouped"]["bound"] == pytest.approx(
        .20101645831539106, abs=1e-10
    )
    assert first["pinball"]["grouped"]["bound"] == pytest.approx(
        2.4730831794305517, abs=1e-10
    )
