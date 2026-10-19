import numpy as np
import pytest

from horos.represent.grouping import nearest_group_hierarchy
from horos.represent.pairing import global_nearest_pairs
from pythia.core.mae_family_envelope import (
    decision_pair_group_envelope, geometry_routed_decision_envelope,
)


@pytest.mark.parametrize("loss,extra", [
    ("pinball", {"quantile": .9, "outcome_range_bound": 100.}),
    ("window_miss", {"window": 10.}),
])
def test_decision_family_joint_selection_and_monotone_smoothness(loss, extra):
    x = np.linspace(0, 1, 64)[:, None]
    y = 40 * x[:, 0] + np.tile([-20., 20.], 32)
    e = np.arange(64)
    pair = global_nearest_pairs(x, e)
    groups = nearest_group_hierarchy(x, e, max_group_size=8)
    args = (x, y, e, pair.left_indices, pair.right_indices, groups)
    low = decision_pair_group_envelope(
        *args, loss=loss, lipschitz_bound=40., **extra,
    )
    high = decision_pair_group_envelope(
        *args, loss=loss, lipschitz_bound=100., **extra,
    )
    assert low["status"] == high["status"] == "complete"
    assert low["confidence"] == pytest.approx(.95)
    assert low["bound"] == max(low["family_bounds"].values())
    assert low["bound"] >= high["bound"]
    assert low["point_estimate_is_certified"] is False
    if loss == "window_miss":
        assert low["attainable_within_window_percentage_upper"] == pytest.approx(
            100 * (1 - low["bound"])
        )


def test_decision_family_refuses_unsupported_loss():
    result = decision_pair_group_envelope(
        np.zeros((4, 1)), np.zeros(4), np.arange(4),
        np.array([0, 2]), np.array([1, 3]), {2: np.arange(4).reshape(2, 2)},
        loss="log_loss", lipschitz_bound=0.,
    )
    assert result["status"] == "refused"


@pytest.mark.parametrize("loss,extra", [
    ("pinball", {"quantile": .1, "outcome_range_bound": 100.}),
    ("window_miss", {"window": 10.}),
])
def test_predeclared_family_policy_spends_no_alpha_on_unused_family(loss, extra):
    x = np.linspace(0, 1, 64)[:, None]
    y = 40 * x[:, 0] + np.tile([-20., 20.], 32)
    e = np.arange(64)
    pair = global_nearest_pairs(x, e)
    groups = nearest_group_hierarchy(x, e, max_group_size=8)
    args = (x, y, e, pair.left_indices, pair.right_indices, groups)
    adaptive = decision_pair_group_envelope(
        *args, loss=loss, lipschitz_bound=40., **extra,
    )
    paired = decision_pair_group_envelope(
        *args, loss=loss, lipschitz_bound=40., family_policy="paired", **extra,
    )
    grouped = decision_pair_group_envelope(
        *args, loss=loss, lipschitz_bound=40., family_policy="grouped", **extra,
    )
    assert paired["family_bounds"] == {"paired": paired["bound"]}
    assert grouped["family_bounds"] == {"grouped": grouped["bound"]}
    assert paired["alpha_allocation_across_families"] == {
        "paired": .05, "grouped": 0.,
    }
    assert grouped["alpha_allocation_across_families"] == {
        "paired": 0., "grouped": .05,
    }
    assert paired["bound"] >= adaptive["family_bounds"]["paired"]
    assert grouped["bound"] >= adaptive["family_bounds"]["grouped"]
    assert paired["simultaneous_across_families"] is False
    assert grouped["simultaneous_across_families"] is False


def test_family_policy_rejects_unknown_route():
    with pytest.raises(ValueError, match="family_policy"):
        decision_pair_group_envelope(
            np.zeros((4, 1)), np.zeros(4), np.arange(4),
            np.array([0, 2]), np.array([1, 3]),
            {2: np.arange(4).reshape(2, 2)},
            loss="window_miss", window=10., lipschitz_bound=0.,
            family_policy="outcome_tuned",
        )


@pytest.mark.parametrize("dimension,expected_family", [
    (1, "grouped"), (4, "paired"),
])
def test_geometry_route_uses_features_and_declared_caps_only(dimension, expected_family):
    rng = np.random.default_rng(41)
    x = (np.linspace(0, 1, 64)[:, None] if dimension == 1
         else rng.uniform(size=(64, dimension)))
    y = 40 * x[:, 0] + rng.uniform(-20, 20, size=64)
    entities = np.arange(64)
    pair = global_nearest_pairs(x, entities)
    groups = nearest_group_hierarchy(x, entities, max_group_size=8)
    result = geometry_routed_decision_envelope(
        x, y, entities, pair.left_indices, pair.right_indices, groups,
        loss="pinball", quantile=.1, outcome_range_bound=100.,
        lipschitz_bound=300. if dimension == 4 else 40.,
    )
    assert result["status"] == "complete"
    assert result["selected_family"] == expected_family
    assert result["geometry_route_outcome_blind"] is True
    assert result["claim_tier"] == (
        "exploratory" if expected_family == "grouped" else "conditional"
    )
