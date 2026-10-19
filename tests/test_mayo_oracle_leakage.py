import pytest
from pathlib import Path

from horos.benchmark.mayo_oracle_leakage import _inject_and_extract, run_mayo_oracle_leakage


DERIVED = Path(__file__).resolve().parents[1] / "data/derived/mayo"


def test_oracle_note_marker_round_trips_without_persisting_a_note():
    assert _inject_and_extract("synthetic pre-cutoff note", 123.5) == 123.5


def test_oracle_note_marker_refuses_preexisting_duplicate_marker():
    with pytest.raises(ValueError, match="uniquely"):
        _inject_and_extract("[PYTHIA-ORACLE] actual duration: 42 minutes", 123.5)


@pytest.mark.skipif(
    not (DERIVED / "note_bundles_recut.parquet").exists(),
    reason="private Mayo notes unavailable",
)
def test_private_mayo_augmentation_falsification_is_logical():
    result = run_mayo_oracle_leakage(DERIVED, n_patients=2048)
    assert result["claim_tier"] == "refused"
    assert result["baseline_frozen_oof_mae_on_selected_patients"] == pytest.approx(
        32.19049765728414, abs=1e-10,
    )
    arms = result["arms"]
    assert [arm["n_injected"] for arm in arms] == [205, 1024, 2048]
    assert [arm["mixed_oracle_plus_frozen_oof_mae"] for arm in arms] == pytest.approx(
        [29.01180959865451, 16.850114420056343, 0.], abs=1e-10,
    )
    assert all(arm["oracle_prediction_mae_on_injected"] == 0 for arm in arms)
    assert arms[-1]["full_oracle_pair_bound_with_L_900"] == 0
    assert arms[-1]["full_oracle_pair_estimate_with_L_900"] == pytest.approx(0., abs=1e-10)
    assert arms[-1]["public_api_status"] == "refused"
    duplicate = result["deterministic_feature_duplication_control"]
    assert duplicate["true_information_change"].startswith("none")
    assert duplicate["metrics"]["original"]["pseudo_bound_at_unaudited_L_300"] == pytest.approx(
        22.214815985098134, abs=1e-10,
    )
    assert duplicate["metrics"]["twenty_extra_schedule_copies"]["pseudo_bound_at_unaudited_L_300"] == pytest.approx(
        20.28808951785879, abs=1e-10,
    )
