import pytest

from pythia.core.decision_order import monotone_window_floor_closure


def test_window_miss_floor_closure_is_tightening_and_ordered():
    closed = monotone_window_floor_closure({15.: .1, 30.: .2, 60.: .05})
    assert closed == {15.: .2, 30.: .2, 60.: .05}
    assert list(closed.values()) == sorted(closed.values(), reverse=True)


@pytest.mark.parametrize("bad", [
    {}, {0.: .1}, {15.: -.1}, {15.: 1.1}, {float("nan"): .2},
])
def test_window_miss_floor_closure_rejects_invalid_registry(bad):
    with pytest.raises(ValueError):
        monotone_window_floor_closure(bad)
