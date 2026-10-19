import numpy as np
import pytest
from pathlib import Path

from horos.benchmark.mayo_text_augmentation import (
    evaluate_text_controls, run_mayo_text_augmentation,
)
from horos.represent.perturb import TextPerturbationSpec


def test_train_free_text_controls_are_aggregate_deterministic_and_same_cohort():
    n = 16
    x = np.linspace(0, 1, n)[:, None]
    notes = np.array([f"[NOTE_TYPE: A] planned ablation region {i % 3}" for i in range(n)])
    y = np.linspace(20, 100, n)
    controls = (
        TextPerturbationSpec("identity", 0, 9),
        TextPerturbationSpec("case_dropout", 1, 9),
    )
    args = (x, notes, y, np.array([f"p{i}" for i in range(n)]),
            np.array([f"case{i}" for i in range(n)]))
    first = evaluate_text_controls(*args, controls=controls, lipschitz_bound=100)
    repeat = evaluate_text_controls(*args, controls=controls, lipschitz_bound=100)
    assert first == repeat
    assert first["pythia_trained_models"] is False
    assert first["text_transform_fitted"] is False
    assert first["controls"]["case_dropout_1"]["empty_after"] == n
    assert "planned ablation" not in str(first)
    for control in first["controls"].values():
        assert set(control["losses"]) == {"pinball_q90", "w30_miss"}
        assert all(row["claim_tier"] == "exploratory" for row in control["losses"].values())


def test_augmentation_refuses_invalid_entities_and_uncovered_outcomes():
    n = 4
    args = (np.zeros((n, 1)), np.array(["note"] * n), np.arange(n, dtype=float),
            np.array(["same"] * n), np.arange(n))
    with pytest.raises(ValueError, match="unique"):
        evaluate_text_controls(*args)
    args = (args[0], args[1], np.array([1., 2., 3., 901.]),
            np.array(["a", "b", "c", "d"]), args[4])
    with pytest.raises(ValueError, match="span"):
        evaluate_text_controls(*args)


DERIVED = Path(__file__).resolve().parents[1] / "data/derived/mayo"


@pytest.mark.skipif(
    not (DERIVED / "note_bundles_recut.parquet").exists(),
    reason="private Mayo note bundles unavailable",
)
def test_private_mayo_train_free_note_controls_are_frozen_aggregate_checks():
    result = run_mayo_text_augmentation(DERIVED)
    assert result["claim_tier"] == "exploratory"
    assert result["pythia_trained_models"] is False
    assert result["n_entities"] == 256
    assert result["n_with_text"] == 216
    rows = result["controls"]
    assert rows["identity"]["changed_rows"] == 0
    assert rows["case_dropout_1"]["empty_after"] == 256
    assert rows["identity"]["losses"]["pinball_q90"]["bound"] == pytest.approx(
        1.6756, abs=1e-4,
    )
    assert rows["identity"]["losses"]["w30_miss"]["bound"] == pytest.approx(
        .2574, abs=1e-4,
    )
    assert rows["case_dropout_1"]["losses"]["w30_miss"]["bound"] < rows[
        "identity"]["losses"]["w30_miss"]["bound"]
