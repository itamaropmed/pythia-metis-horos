import numpy as np
import pytest

from horos.represent.grouping import nearest_group_hierarchy
from horos.represent.pairing import global_nearest_pairs
from pythia.core.mae_family_envelope import mae_pair_group_envelope


def test_pair_group_selection_pays_for_both_families():
    x = np.linspace(0, 1, 64)[:, None]
    y = x[:, 0] + np.tile([-.4, .4], 32)
    entity = np.arange(64)
    pairs = global_nearest_pairs(x, entity)
    groups = nearest_group_hierarchy(x, entity, max_group_size=8)
    result = mae_pair_group_envelope(
        x, y, entity, pairs.left_indices, pairs.right_indices, groups,
        lipschitz_bound=1., outcome_term_clips=(.1, .2, .4, .8),
    )
    assert result["status"] == "complete"
    assert result["confidence"] == pytest.approx(.95)
    assert result["bound"] == max(result["family_bounds"].values())
    assert result["alpha_allocation_across_families"] == {
        "paired": .025, "grouped": .025,
    }
    assert result["simultaneous_across_families"] is True
    assert result["point_estimate_is_certified"] is False


def test_pair_group_refuses_partial_pairing():
    x = np.linspace(0, 1, 32)[:, None]
    entity = np.arange(32)
    groups = nearest_group_hierarchy(x, entity, max_group_size=4)
    result = mae_pair_group_envelope(
        x, x[:, 0], entity, np.arange(0, 32, 2),
        np.r_[np.arange(1, 31, 2), 0], groups,
        lipschitz_bound=1., outcome_term_clips=(.1,),
    )
    assert result["status"] == "refused"


def test_pair_group_bound_weakens_with_larger_smoothness_cap():
    x = np.linspace(0, 1, 64)[:, None]
    y = np.tile([-.5, .5], 32)
    entity = np.arange(64)
    pairs = global_nearest_pairs(x, entity)
    groups = nearest_group_hierarchy(x, entity, max_group_size=8)
    values = [mae_pair_group_envelope(
        x, y, entity, pairs.left_indices, pairs.right_indices, groups,
        lipschitz_bound=l, outcome_term_clips=(.1, .2, .4, .8),
    )["bound"] for l in (0., 1., 10.)]
    assert values[0] >= values[1] >= values[2]


def test_pair_group_zero_bound_is_flagged_uninformative():
    x = np.linspace(0, 1, 32)[:, None]
    entity = np.arange(32)
    pairs = global_nearest_pairs(x, entity)
    groups = nearest_group_hierarchy(x, entity, max_group_size=4)
    result = mae_pair_group_envelope(
        x, np.zeros(32), entity, pairs.left_indices, pairs.right_indices,
        groups, lipschitz_bound=10., outcome_term_clips=(.1, .2),
    )
    assert result["bound"] == 0
    assert result["diagnostic_status"] == "uninformative_at_declared_metric"
