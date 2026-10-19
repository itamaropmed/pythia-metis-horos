import numpy as np

from horos.models.case_risk import fit_model_risk_forecaster


def test_crossfitted_forecaster_orders_unseen_risk_without_test_outcomes():
    rng = np.random.default_rng(812)
    n = 3200
    x = rng.normal(size=(n, 6))
    scale = 8 + 18 / (1 + np.exp(-2 * x[:, 0])) + 10 * (x[:, 1] > 0)
    error = np.abs(rng.normal(0, scale))
    groups = np.arange(n)
    model = fit_model_risk_forecaster(
        x[:2400], error[:2400], groups[:2400], model_name="frozen_model",
        folds=4, seed=10,
    )
    risk, probability = model.predict(x[2400:])
    assert np.corrcoef(risk, error[2400:])[0, 1] > .25
    assert np.all(risk >= 0)
    assert np.all((probability >= 0) & (probability <= 1))


def test_forecaster_rejects_misaligned_inputs():
    x = np.ones((20, 2))
    try:
        fit_model_risk_forecaster(x, np.ones(19), np.arange(20), model_name="bad")
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("misaligned risk targets must be rejected")
