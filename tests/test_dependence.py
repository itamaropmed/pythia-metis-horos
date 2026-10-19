import numpy as np
import pytest

from pythia.core.bregman import bregman_bound
from pythia.core.dependence import dependence_profile
from pythia.suite.dependence_runner import run_dependence_suite


def test_profile_is_monotone_and_matches_zero_rung():
    rng = np.random.default_rng(42)
    m = rng.normal(size=3000)
    y = m + rng.normal(scale=0.5, size=m.size)
    p = np.vstack([
        m + rng.normal(scale=0.2, size=m.size),
        m + rng.normal(scale=0.3, size=m.size),
        m + rng.normal(scale=0.4, size=m.size),
    ])
    result = dependence_profile(y, p, np.array([0.0, 0.2, 0.5, 0.8]))
    bounds = np.array([row["bound"] for row in result["curve"]])
    assert bounds[0] == pytest.approx(bregman_bound(y, p, "mse")["bound"])
    assert np.all(np.diff(bounds) <= 1e-12)
    assert result["status"] == "exploratory"
    assert result["used_for_certificate"] is False
    assert result["feasible_dependence_interval"][0] >= 0
    assert 0 <= result["lambda_break"] <= 1
    assert all("active_constraints" in row for row in result["curve"])


def test_profile_validates_lambda_grid():
    y = np.arange(10.0)
    p = np.vstack([y, y + 1])
    with pytest.raises(ValueError):
        dependence_profile(y, p, np.array([0.4, 0.2]))


def test_latent_common_bias_suite_has_no_population_violations():
    result = run_dependence_suite(seed=7, n=3000)
    assert result["n_designs"] == 24
    assert result["violations"] == 0
