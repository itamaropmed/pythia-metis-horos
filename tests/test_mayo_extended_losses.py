from pathlib import Path

import pytest

from horos.benchmark.mayo_extended_losses import (
    MAYO_DECISIONS, run_mayo_extended_decision_losses,
)


DERIVED = Path(__file__).resolve().parents[1] / "data/derived/mayo"


@pytest.mark.skipif(not (DERIVED / "PB.npy").exists(), reason="private Mayo OOF unavailable")
def test_private_mayo_extended_decision_audit_is_aggregate_and_logical():
    result = run_mayo_extended_decision_losses(DERIVED)
    assert result["claim_tier"] == "exploratory"
    assert result["n_unique_patients"] == 10532
    assert result["n_used_patients"] == 10528
    assert len(result["periods"]) == 4
    assert len(result["loss_registry"]) == len(MAYO_DECISIONS) == 5
    first = result["periods"][0]["representations"]["tabular_metric"]
    assert first["q10_pinball"]["bound"] == pytest.approx(2.4763613776079842, abs=1e-10)
    assert first["q90_pinball"]["bound"] == pytest.approx(4.491383750078989, abs=1e-10)
    assert first["w30_miss"]["bound"] == pytest.approx(.21096865571836723, abs=1e-10)
    assert first["absolute_error_p75_strict_floor_minutes"] == 15.
    assert first["absolute_error_p90_strict_floor_minutes"] == 30.
    for panel in result["periods"]:
        for representation in panel["representations"].values():
            for label in result["loss_registry"]:
                row = representation[label]
                assert row["claim_tier"] == "exploratory"
                assert row["lower_bound_le_achieved_check"] is True
                if label.startswith("w"):
                    assert row["attainable_within_window_percentage_upper"] >= row[
                        "achieved_within_window_percentage"
                    ]


@pytest.mark.skipif(not (DERIVED / "PB.npy").exists(), reason="private Mayo OOF unavailable")
def test_private_mayo_geometry_route_preserves_frozen_strength_exploratorily():
    result = run_mayo_extended_decision_losses(
        DERIVED, window_policy="geometry_route",
    )
    assert result["claim_tier"] == "exploratory"
    assert result["pythia_trained_predictive_model"] is False
    tabular = result["weighted_summary"]["tabular_metric"]
    assert tabular["w30_miss"]["bound"] == pytest.approx(.2678128, abs=1e-5)
    assert tabular["q10_pinball"]["bound"] > 2.37
    note = result["weighted_summary"]["tabular_plus_note_values_and_applicability"]
    assert note["w30_miss"]["bound"] == pytest.approx(.262589, abs=1e-5)
    for panel in result["periods"]:
        for representation in panel["representations"].values():
            for label in result["loss_registry"]:
                assert representation[label]["lower_bound_le_achieved_check"] is True
