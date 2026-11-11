import numpy as np
import pytest

from pythia.core.tight_bounds import (betting_lower_mean, scalar_cert, tune_cert, crossfit_certificate, centre_within,
                                      lam_from_pilot)


def test_betting_covers_nonconstant_means():
    rng = np.random.default_rng(0); miss = 0; reps = 400
    for _ in range(reps):
        mu = rng.uniform(.05, .95, 300)                      # heterogeneous conditional means
        x = (rng.random(300) < mu).astype(float)
        miss += betting_lower_mean(x, .1, .5, seed=1) > mu.mean()
    assert miss / reps <= .1 + .03


def test_betting_validates_inputs():
    with pytest.raises(ValueError):
        betting_lower_mean(np.array([.2, 1.5]), .05, .5)
    with pytest.raises(ValueError):
        betting_lower_mean(np.array([.2, .5]), .05, 1.2)
    with pytest.raises(ValueError):
        betting_lower_mean(np.array([.2, .5]), 1.5, .5)


def _world(n, signal, rng):
    g = rng.integers(0, 5, n).astype(str); s = rng.normal(0, 1, n); m = signal * s
    r = m + rng.normal(0, 20, n); h = centre_within(s + rng.normal(0, .5, n), g)
    return g, m, r, h, s


def test_scalar_cert_refuses_uncentred_and_is_valid():
    rng = np.random.default_rng(1)
    g, m, r, h, _ = _world(2000, 12., rng)
    with pytest.raises(ValueError):
        scalar_cert(h + 1., r, 60., .05, groups=g)
    truth = np.mean(centre_within(np.clip(m, -60, 60), g) ** 2)
    for meth in ("mp", "betting"):
        c = scalar_cert(h, r, 60., .05, method=meth, lam=.5, groups=g)
        assert 0 <= c.bound <= truth * 1.05


def test_betting_tighter_than_mp_and_null_is_quiet():
    rng = np.random.default_rng(2); bet = mp = 0.; claims = 0
    for k in range(30):
        g, m, r, h, _ = _world(1500, 12., rng)
        cfg = tune_cert(h, r, g, .05, len(r), caps=(60.,))
        bet += scalar_cert(h, r, 60., .05, method="betting", lam=cfg["lam"], groups=g).bound
        mp += scalar_cert(h, r, 60., .05, method="mp", groups=g).bound
        g0, _, r0, h0, _ = _world(1500, 0., rng)
        claims += scalar_cert(h0, r0, 60., .05, method="betting", lam=.5, groups=g0).bound > 0
    assert bet > mp
    assert claims <= 4


def test_crossfit_certificate_runs_and_is_bounded():
    rng = np.random.default_rng(3); n = 2400
    g, m, r, h, s = _world(n, 12., rng); X = np.column_stack([s + rng.normal(0, .5, n), rng.normal(0, 1, (n, 3))])

    def train(Xtr, rtr, gtr, Xev, gev):
        b = np.linalg.lstsq(Xtr, centre_within(rtr, gtr), rcond=None)[0]; return Xev @ b
    out = crossfit_certificate(train, X, r, g, .05, K=2, cap=60.)
    assert 0 <= out["bound"] <= np.mean(centre_within(np.clip(m, -60, 60), g) ** 2) * 1.1
    assert len(out["folds"]) == 2 and lam_from_pilot(.01, 1000, .05) <= .9
