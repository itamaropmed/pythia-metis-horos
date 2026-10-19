"""Population-level finite enumerations for pair and group decision witnesses."""
from itertools import product

import numpy as np
import pytest

from pythia.core.group_witness import grouped_decision_lower_bound
from pythia.core.replicates import paired_replicate_lower_bound


@pytest.mark.parametrize("center_rule", ["mean", "coordinate_median"])
@pytest.mark.parametrize("loss,extra,true_risk", [
    ("pinball", {"quantile": .1}, 2.),
    ("pinball", {"quantile": .9}, 2.),
    ("window_miss", {"window": 10.}, .5),
])
def test_exact_independent_two_point_laws_obey_witness_inequality(
    loss, extra, true_risk, center_rule,
):
    x = np.array([0., .2, .8, 1.])[:, None]
    centers = 30 * x[:, 0]
    groups = np.array([[0, 1], [2, 3]])
    pair_left, pair_right = groups[:, 0], groups[:, 1]
    pair_estimates, group_estimates = [], []
    for signs in product((-1., 1.), repeat=4):
        y = centers + 20 * np.asarray(signs)
        paired = paired_replicate_lower_bound(
            y[pair_left], y[pair_right], np.array([.2, .2]),
            pair_left, pair_right, np.arange(2), loss=loss,
            lipschitz_bound=30., distance_bound=1., alpha=.05,
            outcome_difference_bound=100. if loss == "pinball" else None,
            concentration="conditional_bentkus", **extra,
        )
        grouped = grouped_decision_lower_bound(
            x, y, np.arange(4), groups, loss=loss,
            lipschitz_bound=30., alpha=.05,
            outcome_range_bound=100. if loss == "pinball" else None,
            concentration="conditional_bentkus", center_rule=center_rule, **extra,
        )
        assert paired["status"] == grouped["status"] == "complete"
        pair_estimates.append(paired["estimate"])
        group_estimates.append(grouped["estimate"])
    assert np.mean(pair_estimates) <= true_risk + 1e-12
    assert np.mean(group_estimates) <= true_risk + 1e-12


def test_sharp_group_pinball_range_recovers_q50_half_mae():
    x = np.zeros((4, 1))
    y = np.array([-20., 20., -20., 20.])
    groups = np.array([[0, 1], [2, 3]])
    shared = dict(
        representation=x, outcomes=y, entities=np.arange(4), groups=groups,
        lipschitz_bound=0., outcome_range_bound=40., alpha=.05,
        concentration="conditional_bentkus",
    )
    mae = grouped_decision_lower_bound(**shared, loss="mae")
    q50 = grouped_decision_lower_bound(**shared, loss="pinball", quantile=.5)
    assert q50["estimate"] == pytest.approx(.5 * mae["estimate"])
    assert q50["bound"] == pytest.approx(.5 * mae["bound"])


def test_group_pinball_cap_is_attained_by_two_atom_extremal_mix():
    x = np.zeros((20, 1))
    y = np.tile(np.array([-20.] + [20.] * 9), 2)
    result = grouped_decision_lower_bound(
        x, y, np.arange(20), np.arange(20).reshape(2, 10),
        loss="pinball", quantile=.1, lipschitz_bound=0.,
        outcome_range_bound=40., alpha=.05,
        concentration="conditional_bentkus",
    )
    assert result["estimate"] == pytest.approx(.1 * .9 * 40.)


@pytest.mark.parametrize("size,expected_factor", [(2, .05), (4, .075), (10, .09)])
def test_finite_group_pinball_range_is_sharp_for_q10(size, expected_factor):
    x = np.zeros((2 * size, 1))
    low_count = 1
    y = np.tile(np.array([-20.] * low_count + [20.] * (size - low_count)), 2)
    result = grouped_decision_lower_bound(
        x, y, np.arange(len(y)), np.arange(len(y)).reshape(2, size),
        loss="pinball", quantile=.1, lipschitz_bound=0.,
        outcome_range_bound=40., concentration="conditional_bentkus",
        pinball_range_rule="finite_group",
    )
    assert result["finite_group_pinball_range_factor"] == pytest.approx(expected_factor)
    assert result["score_range"] == pytest.approx(expected_factor * 40)
    assert result["estimate"] <= expected_factor * 40 + 1e-12
    if size in {2, 4, 10}:
        assert result["estimate"] == pytest.approx(expected_factor * 40)


@pytest.mark.parametrize("loss,extra", [
    ("mae", {}), ("pinball", {"quantile": .1}),
    ("pinball", {"quantile": .9}), ("window_miss", {"window": 10.}),
])
def test_exact_lipschitz_group_witness_dominates_radial_relaxation(loss, extra):
    x = np.array([0., .07, .4, 1., .02, .35, .7, .98])[:, None]
    y = np.array([10., 38., 30., 60., 45., 13., 67., 26.])
    common = dict(
        representation=x, outcomes=y, entities=np.arange(8),
        groups=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        loss=loss, lipschitz_bound=20.,
        outcome_range_bound=100. if loss != "window_miss" else None,
        concentration="conditional_bentkus", **extra,
    )
    radial = grouped_decision_lower_bound(**common, witness_rule="radial")
    exact = grouped_decision_lower_bound(**common, witness_rule="exact_lipschitz")
    assert exact["status"] == "complete"
    assert exact["claim_tier"] == (
        "conditional" if loss == "window_miss" else "exploratory"
    )
    assert exact["estimate"] >= radial["estimate"] - 1e-9
    assert exact["bound"] >= radial["bound"] - 1e-9


@pytest.mark.parametrize("tau,true_risk", [(.1, 2.), (.9, 2.)])
def test_exact_lipschitz_group_witness_obeys_population_inequality(tau, true_risk):
    x = np.array([0., .1, .4, 1.])[:, None]
    centers = 30 * x[:, 0]
    group = np.arange(4).reshape(1, 4)
    # Two groups are required by the public concentration API.
    xx = np.tile(x, (2, 1))
    groups = np.arange(8).reshape(2, 4)
    scores = []
    for signs in product((-1., 1.), repeat=8):
        y = np.tile(centers, 2) + 20 * np.asarray(signs)
        result = grouped_decision_lower_bound(
            xx, y, np.arange(8), groups, loss="pinball", quantile=tau,
            lipschitz_bound=30., outcome_range_bound=100.,
            concentration="conditional_bentkus", witness_rule="exact_lipschitz",
        )
        assert result["status"] == "complete"
        scores.append(result["estimate"])
    assert np.mean(scores) <= true_risk + 1e-10


def test_exact_lipschitz_window_miss_population_inequality():
    x = np.array([0., .1, .4, 1.])[:, None]
    centers = 30 * x[:, 0]
    xx = np.tile(x, (2, 1))
    groups = np.arange(8).reshape(2, 4)
    scores = []
    for signs in product((-1., 1.), repeat=8):
        y = np.tile(centers, 2) + 20 * np.asarray(signs)
        result = grouped_decision_lower_bound(
            xx, y, np.arange(8), groups, loss="window_miss", window=10.,
            lipschitz_bound=30., concentration="conditional_bentkus",
            witness_rule="exact_lipschitz",
        )
        assert result["status"] == "complete"
        scores.append(result["estimate"])
    assert np.mean(scores) <= .5 + 1e-10


def test_exact_lipschitz_window_refuses_unbounded_group_size():
    result = grouped_decision_lower_bound(
        np.zeros((50, 1)), np.arange(50.), np.arange(50),
        np.arange(50).reshape(2, 25), loss="window_miss", window=1.,
        lipschitz_bound=0., witness_rule="exact_lipschitz",
    )
    assert result["status"] == result["claim_tier"] == "refused"


def test_entity_bounded_differences_window_radius_and_one_case_sensitivity():
    x = np.zeros((32, 1))
    y = np.tile(np.array([0., 30., 60., 90.]), 8)
    groups = np.arange(32).reshape(2, 16)
    common = dict(
        representation=x, entities=np.arange(32), groups=groups,
        loss="window_miss", window=10., lipschitz_bound=0.,
        witness_rule="exact_lipschitz", concentration="conditional_entity_mcd",
        alpha=.05,
    )
    baseline = grouped_decision_lower_bound(outcomes=y, **common)
    altered = y.copy()
    altered[0] = 90.
    changed = grouped_decision_lower_bound(outcomes=altered, **common)
    assert baseline["radius"] == pytest.approx(np.sqrt(np.log(20) / 64))
    assert abs(baseline["estimate"] - changed["estimate"]) <= 1 / 32 + 1e-12


def test_entity_bounded_differences_refuses_clipped_unbounded_pinball():
    result = grouped_decision_lower_bound(
        np.zeros((4, 1)), np.arange(4.), np.arange(4),
        np.arange(4).reshape(2, 2), loss="pinball", quantile=.1,
        lipschitz_bound=0., outcome_term_clip=2.,
        concentration="conditional_entity_mcd",
    )
    assert result["status"] == result["claim_tier"] == "refused"
