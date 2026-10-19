import numpy as np
from pythia.core.global_bound import joint_floor_upper, table_floor_bounds, certified_easy_threshold


def test_any_loss_upper_and_table_bounds_valid():
    r = np.random.default_rng(0); n = 3000; cell = r.integers(0, 5, n); E = r.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(.5 * (cell - 2) + 1.5 * (E[:, 0] > 0) - .75))); y = (r.random(n) < p).astype(int)
    pbar = np.array([p[cell == c].mean() for c in cell])
    for L, f in (("01", lambda q: np.minimum(q, 1 - q)), ("brier", lambda q: q * (1 - q)),
                 ("log", lambda q: -(q * np.log(q) + (1 - q) * np.log(1 - q)))):
        assert joint_floor_upper(L, cell, E, y, 2)["upper"] >= f(p).mean()
        tb = table_floor_bounds(L, cell, y, 2)
        assert tb["lower"] <= f(pbar).mean() <= tb["upper"]


def test_easy_threshold_controls_error():
    r = np.random.default_rng(1); s = r.random(4000); err = (r.random(4000) < .05 + .5 * s).astype(float)
    t = certified_easy_threshold(s, err, eps=.15, delta=.05)
    assert t is not None and (.05 + .5 * s[s <= t]).mean() <= .15
