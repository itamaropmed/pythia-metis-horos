import numpy as np
import pytest

from pythia import dkelly_lower_mean, truncated_saa_portfolio


def test_dkelly_basic_limits():
    assert dkelly_lower_mean(np.zeros(20), cap=1., alpha=.05) == 0.
    value = dkelly_lower_mean(np.full(500, 2.), cap=2., alpha=.05)
    assert 0 < value <= 2.


def test_portfolio_is_original_scale_and_selection_safe():
    result = truncated_saa_portfolio(
        {"k2": np.full(300, 1.), "k4": np.full(150, 1.2)},
        loss="MAE", score_caps=(.5, 1., 2.), alpha_family=.05,
        retained_fractions={"k2": 1., "k4": .8},
        route_provenance={"block_registry": "sha256:abc"},
        portfolio_frozen_before_evaluation=True, outcome_scale="minutes")
    assert result.status == "conditional"
    assert 0 < result.bound <= 1.2
    assert result.alpha_per_candidate == pytest.approx(.05 / 6)
    assert result.diagnostics["outcome_clipping_used"] is False
    assert result.diagnostics["score_truncation_used"] is True
    assert result.diagnostics["moment_assumption_required"] is False


def test_portfolio_refuses_missing_provenance_and_posthoc_selection():
    missing = truncated_saa_portfolio(
        {"k2": [1., 2.]}, loss="MAE", score_caps=(1.,),
        route_provenance=None, portfolio_frozen_before_evaluation=True)
    assert missing.status == "refused"
    posthoc = truncated_saa_portfolio(
        {"k2": [1., 2.]}, loss="MAE", score_caps=(1.,),
        route_provenance={"hash": "x"}, portfolio_frozen_before_evaluation=False)
    assert posthoc.status == "refused"


def test_dkelly_heterogeneous_unbounded_scores_after_safe_truncation():
    # Independent exponentials with different means.  The theorem targets the
    # average mean and does not require iid observations.  Truncation makes the
    # e-value bounded while preserving a lower target for the original mean.
    rng = np.random.default_rng(20261014)
    means = np.linspace(.5, 2., 80)
    truth = float(np.mean(means))
    violations = 0
    repetitions = 4000
    for _ in range(repetitions):
        raw = rng.exponential(means)
        capped = np.minimum(raw, 2.)
        lower = dkelly_lower_mean(capped, cap=2., alpha=.05)
        violations += lower > truth + 1e-12
    assert violations / repetitions < .06
