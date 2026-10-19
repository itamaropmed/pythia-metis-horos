import numpy as np
import pytest
from scipy.stats import binom

from pythia import calibrate_repeated_action_dispersion
from pythia.core.action_dispersion_calibration import _simultaneous_quantile_ranks


def test_order_stat_ranks_spend_no_more_than_registered_tail_budget():
    n, q, alpha, units = 200, 0.9, 0.05, 8
    lower, upper = _simultaneous_quantile_ranks(n, q, alpha, units)
    tail = alpha / (2 * units)
    assert binom.cdf(lower - 1, n, q) <= tail
    assert binom.sf(upper - 1, n, q) <= tail


def test_repeated_action_calibration_contains_known_quantile_dispersion():
    rng = np.random.default_rng(8123)
    locations = np.array([-1.0, -0.3, 0.4, 1.5])
    y = locations[None, :] + rng.normal(size=(2000, 4))
    blocks = np.array([[0, 1, 2, 3]])
    result = calibrate_repeated_action_dispersion(
        y, blocks, quantile=0.5, alpha=0.05, support=(-8, 8),
        environment_provenance="independent-lab-runs-v1",
        support_provenance="physical-range-v1",
        curvature_bounds=np.repeat(2 / np.sqrt(2 * np.pi), 4),
        curvature_provenance="normal-density-envelope-v1",
        independence_attested=True,
    )
    true_variance = float(np.var(locations))
    assert result.action_variance_cap >= true_variance
    assert result.curvature_penalty_cap is not None
    assert result.lower_rank > 0
    assert result.upper_rank <= 2000


def test_action_calibration_refuses_missing_contracts_and_large_blocks():
    y = np.zeros((20, 18))
    with pytest.raises(ValueError, match="attested"):
        calibrate_repeated_action_dispersion(
            y, np.array([[0, 1]]), quantile=.5, alpha=.05, support=(-1, 1),
            environment_provenance="env", support_provenance="support")
    with pytest.raises(ValueError, match="at most 16"):
        calibrate_repeated_action_dispersion(
            y, np.arange(17)[None, :], quantile=.5, alpha=.05, support=(-1, 1),
            environment_provenance="env", support_provenance="support",
            independence_attested=True)


def test_calibration_allows_cross_unit_dependence_within_rows():
    rng = np.random.default_rng(4)
    common = rng.normal(size=(400, 1))
    y = common + np.array([[0.0, 1.0]])
    result = calibrate_repeated_action_dispersion(
        y, np.array([[0, 1]]), quantile=.5, alpha=.1, support=(-6, 7),
        environment_provenance="iid-rows-shared-shock-v1",
        support_provenance="hard-support-v1", independence_attested=True)
    assert result.action_variance_cap >= 0.25
