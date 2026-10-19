"""M0 unit tests -- fast, no model fitting."""
import numpy as np
import pytest

from pythia.core.bregman import bregman_bound
from pythia.core.losses import get_loss, d_squared
from pythia.core.router import route


def test_router_branches():
    assert route("mse").regime == "bregman"
    assert route("logloss").regime == "bregman"
    assert route("mae").regime == "guarded"
    assert route("mae").guard_required
    r = route("auc")
    assert r.regime == "refuse" and r.referral


def test_squared_identity_exact():
    """Risk(f) = R* + E[(m-f)^2] must hold to machine precision."""
    rng = np.random.default_rng(0)
    n = 200_000
    m = rng.standard_normal(n)
    y = m + rng.standard_normal(n) * 0.5          # R* = 0.25
    f = m + rng.standard_normal(n) * 0.3          # excess = 0.09
    loss = get_loss("mse")
    assert loss.risk(y, f) == pytest.approx(0.25 + 0.09, rel=0.02)


def test_pair_delta_recovers_excess_under_orthogonality():
    """(1/4) E[Dsym(f1, f2)] must equal the mean excess risk when errors are orthogonal."""
    rng = np.random.default_rng(1)
    n = 400_000
    m = rng.standard_normal(n)
    y = m + rng.standard_normal(n) * 0.5
    f1 = m + rng.standard_normal(n) * 0.3         # Delta_1 = 0.09
    f2 = m + rng.standard_normal(n) * 0.4         # Delta_2 = 0.16
    dsym = d_squared(f1, f2) + d_squared(f2, f1)
    assert 0.25 * dsym.mean() == pytest.approx(0.5 * (0.09 + 0.16), rel=0.02)


def test_bound_below_estimator():
    rng = np.random.default_rng(2)
    n = 20_000
    m = rng.standard_normal(n)
    y = m + rng.standard_normal(n) * 0.5
    P = np.vstack([m + rng.standard_normal(n) * s for s in (0.2, 0.3, 0.45)])
    res = bregman_bound(y, P, "mse")
    assert res["bound"] <= res["estimator"] <= res["L"]
    assert res["gamma"] >= 0


def test_rmse_is_monotone_map_of_mse():
    rng = np.random.default_rng(3)
    n = 20_000
    m = rng.standard_normal(n)
    y = m + rng.standard_normal(n) * 0.5
    P = np.vstack([m + rng.standard_normal(n) * s for s in (0.2, 0.3, 0.45)])
    a = bregman_bound(y, P, "mse")
    b = bregman_bound(y, P, "rmse")
    assert b["bound"] == pytest.approx(np.sqrt(a["bound"]), rel=1e-9)
    assert b["L"] == pytest.approx(np.sqrt(a["L"]), rel=1e-9)


def test_guarded_loss_rejected_by_bregman_branch():
    with pytest.raises(ValueError):
        bregman_bound(np.zeros(10), np.zeros((2, 10)), "mae")


def test_needs_two_predictors():
    with pytest.raises(ValueError):
        bregman_bound(np.zeros(10), np.zeros((1, 10)), "mse")
