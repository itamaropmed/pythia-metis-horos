import numpy as np
import pytest

from horos.represent.grouping import nearest_group_hierarchy
from pythia.core.group_witness import (
    grouped_decision_envelope, grouped_decision_lower_bound,
)


def test_group_hierarchy_is_deterministic_outcome_free_partition():
    x = np.linspace(0, 1, 32)[:, None]
    entity = np.arange(32)
    first = nearest_group_hierarchy(x, entity, max_group_size=8)
    second = nearest_group_hierarchy(x, entity, max_group_size=8)
    assert set(first) == {2, 4, 8}
    for size, groups in first.items():
        assert groups.shape == (32 // size, size)
        assert np.array_equal(np.sort(groups.ravel()), np.arange(32))
        assert np.array_equal(groups, second[size])


def test_grouped_window_and_pinball_exact_centers_without_penalty():
    x = np.zeros((8, 1))
    y = np.array([0., 0., 100., 100.] * 2)
    entity = np.arange(8)
    groups = np.arange(8).reshape(2, 4)
    miss = grouped_decision_lower_bound(
        x, y, entity, groups, loss="window_miss", window=10.,
        lipschitz_bound=0., alpha=.05,
    )
    pinball = grouped_decision_lower_bound(
        x, y, entity, groups, loss="pinball", quantile=.9,
        outcome_range_bound=100., lipschitz_bound=0., alpha=.05,
    )
    assert miss["estimate"] == pytest.approx(.5)
    assert pinball["estimate"] == pytest.approx(5.)
    assert 0 <= miss["bound"] <= miss["estimate"]
    assert 0 <= pinball["bound"] <= pinball["estimate"]


def test_grouped_witness_weakens_with_larger_smoothness_cap():
    x = np.linspace(0, 1, 32)[:, None]
    y = 100 * x[:, 0] + np.tile([0., 0., 20., 20.], 8)
    groups = nearest_group_hierarchy(x, np.arange(32), max_group_size=4)[4]
    values = [grouped_decision_lower_bound(
        x, y, np.arange(32), groups, loss="window_miss", window=5.,
        lipschitz_bound=l,
    )["bound"] for l in (0., 10., 100.)]
    assert values[0] >= values[1] >= values[2]


def test_grouped_witness_refuses_overlap_and_invalid_raw_cap():
    x = np.zeros((8, 1))
    y = np.array([0., 0., 100., 100.] * 2)
    group = np.arange(8).reshape(2, 4)
    duplicate = group.copy(); duplicate[1, 0] = 0
    assert grouped_decision_lower_bound(
        x, y, np.arange(8), duplicate, loss="window_miss", window=10.,
        lipschitz_bound=0.,
    )["status"] == "refused"
    assert grouped_decision_lower_bound(
        x, y, np.arange(8), group, loss="pinball", quantile=.9,
        outcome_range_bound=99., lipschitz_bound=0.,
    )["status"] == "refused"


def test_grouped_envelope_marks_simultaneous_selection():
    x = np.linspace(0, 1, 32)[:, None]
    y = np.tile([0., 0., 100., 100.], 8)
    hierarchy = nearest_group_hierarchy(x, np.arange(32), max_group_size=8)
    result = grouped_decision_envelope(
        x, y, np.arange(32), hierarchy, loss="window_miss", window=10.,
        lipschitz_bound=0., alpha=.05,
    )
    assert result["status"] == "complete"
    assert result["confidence"] == .95
    assert result["selection_adjusted_within_supplied_registry"] is True
    assert result["registry_outcome_blind_provenance_audited"] is False
    assert result["predeclared_group_sizes"] == [2, 4, 8]


def test_grouped_witness_rejects_fractional_indices():
    with pytest.raises(ValueError, match="integer dtype"):
        grouped_decision_lower_bound(
            np.zeros((8, 1)), np.arange(8.), np.arange(8),
            np.arange(8.).reshape(2, 4) + .5,
            loss="window_miss", window=1., lipschitz_bound=0.,
        )


def test_grouped_mae_exact_score_and_clip_without_raw_outcome_cap():
    x = np.zeros((8, 1))
    y = np.array([0., 0., 100., 100.] * 2)
    groups = np.arange(8).reshape(2, 4)
    plain = grouped_decision_lower_bound(
        x, y, np.arange(8), groups, loss="mae", lipschitz_bound=0.,
        outcome_range_bound=100.,
    )
    clipped = grouped_decision_lower_bound(
        x, y, np.arange(8), groups, loss="mae", lipschitz_bound=0.,
        outcome_term_clip=10.,
    )
    assert plain["estimate"] == pytest.approx(50.)
    assert clipped["estimate"] == pytest.approx(10.)
    assert clipped["outcome_range_bound"] is None
    assert 0 <= clipped["bound"] <= 10.
