import numpy as np
import pytest

from horos.benchmark.decision_plugins import crossfit_local_decision_plugins
from horos.benchmark.multiloss_synthetic import (
    MULTILOSS_DESIGNS, extract_note_concept, multiloss_design,
)


def test_multiloss_registry_has_numeric_and_preoutcome_notes():
    assert len(MULTILOSS_DESIGNS) == len(set(MULTILOSS_DESIGNS)) == 10
    assert sum(name.startswith("note_") for name in MULTILOSS_DESIGNS) == 5
    assert extract_note_concept("preop: no prior implant", "note_negation") == 0
    assert extract_note_concept("preop: prior implant present", "note_negation") == 1


@pytest.mark.parametrize("name", MULTILOSS_DESIGNS)
def test_analytic_multiloss_risks_and_action_caps(name):
    design = multiloss_design(np.random.default_rng(804), name, 100_000)
    x, y = design["representation"], design["outcomes"]
    center, scale = design["center"], design["noise_scale"]
    assert len(design["preoutcome_notes"] or []) == (len(y) if name.startswith("note_") else 0)
    for tau, key in ((.1, "truth_q10_pinball"), (.9, "truth_q90_pinball")):
        action = center + (2 * tau - 1) * scale
        residual = y - action
        observed = np.mean(np.maximum(tau * residual, (tau - 1) * residual))
        assert observed == pytest.approx(np.mean(design[key]), abs=.08)
        left = np.arange(0, 1000, 2)
        right = left + 1
        distances = np.mean(np.abs(x[left] - x[right]), axis=1)
        assert np.all(np.abs(action[left] - action[right])
                      <= design["lipschitz_cap"] * distances + 1e-12)
    for width in (15, 30, 60):
        observed_miss = np.mean(np.abs(y - center) > width)
        assert observed_miss == pytest.approx(
            np.mean(design[f"truth_w{width}_miss"]), abs=.006,
        )


def test_external_plugin_rules_crossfit_without_entering_pythia():
    design = multiloss_design(np.random.default_rng(19), "note_scale", 128)
    result = crossfit_local_decision_plugins(
        design["representation"], design["outcomes"], seed=11,
    )
    assert set(result["methods"]) == {"knn", "histogram", "kernel"}
    assert result["pythia_trained_model"] is False
    for values in result["methods"].values():
        for width in (15, 30, 60):
            assert 0 <= values[f"w{width}_percentage"] <= 100
            assert values[f"w{width}_percentage"] == pytest.approx(
                100 * (1 - values[f"w{width}_miss"])
            )
