import numpy as np

from horos.benchmark.mayo_temporal_stability import _period_pair_bounds


def test_temporal_period_methods_share_pairs_and_keep_bound_invariants():
    rng = np.random.default_rng(19)
    x = rng.uniform(size=(100, 2))
    y = x[:, 0] + rng.uniform(-.5, .5, 100)
    result = _period_pair_bounds(
        x, y, np.arange(100), alpha=.01,
        lipschitz_grid=(0.0, 1.0, 2.0),
    )
    assert result["n_pairs"] == 50
    for method in result["comparison"].values():
        assert all(0 <= row["bound"] <= row["corrected_center"] for row in method)
        assert np.all(np.diff([row["bound"] for row in method]) <= 1e-12)
