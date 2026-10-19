from pathlib import Path

import pytest

from horos.benchmark.mayo_replicate import run_mayo_replicate_diagnostic


DERIVED = Path(__file__).parents[1] / "data" / "derived" / "mayo"


@pytest.mark.skipif(not (DERIVED / "PB.npy").exists(), reason="private Mayo artifacts unavailable")
def test_mayo_replicate_diagnostic_is_aggregate_and_exploratory():
    result = run_mayo_replicate_diagnostic(DERIVED, lipschitz_grid=(0, 400))
    assert result["status"] == "exploratory"
    assert result["pythia_trained_models"] is False
    assert result["n_patients"] == 10532
    assert set(result["representations"]) == {
        "tabular_metric", "tabular_plus_note_values_and_applicability",
    }
    assert "iid" in result["reason_not_certified"]
