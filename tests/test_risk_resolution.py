import numpy as np

from pythia.core.risk_resolution import risk_resolution_diagnostic


def test_exact_conditional_mean_obeys_resolution_identity():
    rng = np.random.default_rng(44)
    z = rng.normal(size=200_000)
    mean = 20 + 8 * z
    loss = mean + rng.normal(0, 6, len(z))
    loss = loss - min(loss.min(), 0)
    mean = mean - min(loss.min(), 0)
    result = risk_resolution_diagnostic(loss, mean)
    truth = np.var(mean) / np.var(loss)
    assert abs(result.squared_pearson_resolution - truth) < .01


def test_constant_forecast_reports_insufficient_resolution():
    result = risk_resolution_diagnostic(np.arange(1., 21.), np.ones(20))
    assert result.status == "insufficient_resolution"
    assert result.squared_pearson_resolution == 0


def test_optional_reference_is_not_a_default_objective_and_is_validated():
    result = risk_resolution_diagnostic(np.arange(1., 21.), np.arange(1., 21.))
    assert result.reference_correlation is None
    assert result.resolution_gap_to_reference is None
    try:
        risk_resolution_diagnostic(np.ones(5), np.ones(5), reference_correlation=1.)
    except ValueError as exc:
        assert "reference correlation" in str(exc)
    else:
        raise AssertionError("invalid reference must be refused")
