import numpy as np
import pytest

from horos.benchmark.omniscient_lp_baseline import global_lipschitz_mae_lp


def test_global_lp_recovers_median_objective_at_zero_lipschitz():
    x = np.linspace(0, 1, 4)[:, None]
    y = np.array([0., 0., 10., 10.])
    result = global_lipschitz_mae_lp(x, y, lipschitz_bound=0.)
    assert result["status"] == "complete"
    assert result["empirical_omniscient_mae"] == pytest.approx(5.)


def test_global_lp_zero_with_enough_smoothness():
    x = np.linspace(0, 1, 4)[:, None]
    y = 2 * x[:, 0]
    result = global_lipschitz_mae_lp(x, y, lipschitz_bound=2.)
    assert result["empirical_omniscient_mae"] == pytest.approx(0.)
