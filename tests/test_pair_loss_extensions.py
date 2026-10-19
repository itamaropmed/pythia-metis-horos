import numpy as np
import pytest

from pythia.core.replicates import (
    paired_replicate_lower_bound, paired_replicate_sensitivity_curve,
)


def _constant_gap_pairs(gap: float, n: int = 1000):
    return (
        np.zeros(n), np.full(n, gap), np.zeros(n),
        np.arange(n), np.arange(n, 2 * n), np.arange(n),
    )


def test_pinball_pair_envelope_has_correct_quantile_scaling():
    result = paired_replicate_lower_bound(
        *_constant_gap_pairs(100.0), loss="pinball", quantile=.9,
        lipschitz_bound=0.0, distance_bound=1.0,
        outcome_difference_bound=100.0,
        concentration="conditional_hoeffding",
    )
    assert np.isclose(result["estimate"], 5.0)
    assert 0 < result["bound"] <= 5.0
    assert "conditional quantile" in result["assumptions"][4]


def test_window_miss_uses_strict_two_window_separation_without_outcome_cap():
    separated = paired_replicate_lower_bound(
        *_constant_gap_pairs(61.0), loss="w30_miss",
        lipschitz_bound=0.0, distance_bound=1.0,
        outcome_difference_bound=None,
        concentration="conditional_hoeffding",
    )
    touching = paired_replicate_lower_bound(
        *_constant_gap_pairs(60.0), loss="w30_miss",
        lipschitz_bound=0.0, distance_bound=1.0,
        outcome_difference_bound=None,
        concentration="conditional_hoeffding",
    )
    assert separated["estimate"] == .5
    assert 0 < separated["bound"] <= .5
    assert touching["bound"] == touching["estimate"] == 0
    assert "without a raw outcome-difference cap" in separated["assumptions"][-1]


def test_window_sensitivity_curve_spends_alpha_over_L_grid():
    args = _constant_gap_pairs(70.0)
    curve = paired_replicate_sensitivity_curve(
        *args, loss="w30_miss", lipschitz_bounds=np.array([0.0, 5.0, 15.0]),
        distance_bound=1.0, outcome_difference_bound=None,
        concentration="conditional_hoeffding",
    )
    direct = paired_replicate_lower_bound(
        *args, loss="w30_miss", lipschitz_bound=0.0,
        distance_bound=1.0, outcome_difference_bound=None,
        alpha=.05 / 3, concentration="conditional_hoeffding",
    )
    assert curve["alpha_allocation"] == "Bonferroni over Lipschitz grid"
    assert np.isclose(curve["curve"][0]["bound"], direct["bound"])


def test_pinball_and_window_inputs_fail_closed():
    with pytest.raises(ValueError, match="quantile"):
        paired_replicate_lower_bound(
            *_constant_gap_pairs(1.0), loss="pinball", lipschitz_bound=0,
            distance_bound=1, outcome_difference_bound=1,
        )
    with pytest.raises(ValueError, match="window"):
        paired_replicate_lower_bound(
            *_constant_gap_pairs(1.0), loss="window_miss", lipschitz_bound=0,
            distance_bound=1, outcome_difference_bound=None,
        )
    with pytest.raises(ValueError, match="outcome-difference cap"):
        paired_replicate_lower_bound(
            *_constant_gap_pairs(1.0), loss="mae", lipschitz_bound=0,
            distance_bound=1, outcome_difference_bound=None,
        )
from pythia.core.replicates import (
    paired_replicate_concentration_envelope, paired_replicate_window_curve,
)


def test_concentration_envelope_allocates_alpha_for_selection():
    import numpy as np

    n = 100
    result = paired_replicate_concentration_envelope(
        np.zeros(n), np.full(n, 100.0), np.zeros(n),
        np.arange(n), np.arange(n, 2 * n), np.arange(n),
        loss="w30_miss", lipschitz_bound=0.0, distance_bound=1.0,
        outcome_difference_bound=None, alpha=.05,
    )
    assert result["status"] == "complete"
    assert result["simultaneous"] is True
    assert result["confidence"] == .95
    assert result["bound"] == max(row["bound"] for row in result["method_rows"])


def test_window_curve_is_simultaneous_and_yields_error_quantile_floor():
    import numpy as np

    n = 1000
    result = paired_replicate_window_curve(
        np.zeros(n), np.full(n, 100.0), np.zeros(n),
        np.arange(n), np.arange(n, 2 * n), np.arange(n),
        windows=np.array([10.0, 20.0, 60.0]),
        lipschitz_bound=0.0, distance_bound=1.0,
        absolute_error_quantiles=(.9,),
    )
    assert result["status"] == "complete"
    assert result["simultaneous"] is True
    assert result["curve"][0]["minimum_miss_risk"] > .1
    assert result["curve"][2]["minimum_miss_risk"] == 0.0
    assert result["absolute_error_quantile_lower_floors"] == [
        {"quantile": .9, "strict_lower_floor": 20.0}
    ]
