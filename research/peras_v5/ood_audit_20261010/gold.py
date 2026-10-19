"""Exact (numerical-integration) Bayes risks for the noise families used in the
benchmark.  Every conditional law is  Y | x  =  mu(x) + s(x) * Z_f  where Z_f is a
standardised family f.  Bayes risks are location-invariant, so for each family we
precompute on a fine grid:

    var_f            MSE*  = s^2 var_f
    mad_f            MAE*  = s   E|Z - med Z|
    pin_f(q)         Q_q*  = s   min_a E rho_q(Z - a)
    win_f(c)         W_h*  = 1 - max_a P(|Z - a| <= c),  c = h / s   (interpolated)

No sampling is involved, so the gold values carry no Monte-Carlo error.
"""
from __future__ import annotations
import numpy as np
from scipy import stats

GRID = np.linspace(-80.0, 80.0, 320001)          # step 5e-4
DZ = GRID[1] - GRID[0]


def _mix(*parts):
    def cdf(z):
        return sum(w * d.cdf(z) for w, d in parts)
    return cdf


FAMILIES = {
    "normal":   _mix((1.0, stats.norm())),
    "laplace":  _mix((1.0, stats.laplace())),                 # b = 1
    "gamma2":   _mix((1.0, stats.gamma(2.0))),                # theta = 1  (pair-twin of laplace)
    "lognorm":  _mix((1.0, stats.lognorm(0.6))),
    "t3":       _mix((1.0, stats.t(3))),
    "t5":       _mix((1.0, stats.t(5))),
    "mix05":    _mix((0.95, stats.norm()), (0.05, stats.norm(3.5, 1.0))),
    "mix35":    _mix((0.65, stats.norm()), (0.35, stats.norm(3.5, 1.0))),
}

C_GRID = np.linspace(0.0, 40.0, 4001)


def _profile(cdf):
    F = cdf(GRID)
    f = np.gradient(F, DZ)
    mean = np.sum(GRID * f) * DZ
    var = np.sum((GRID - mean) ** 2 * f) * DZ
    med = GRID[np.searchsorted(F, 0.5)]
    mad = (np.sum(F[GRID < med]) + np.sum(1 - F[GRID >= med])) * DZ

    def pin(q):
        a = GRID[np.searchsorted(F, q)]
        return (q * np.sum(1 - F[GRID >= a]) + (1 - q) * np.sum(F[GRID < a])) * DZ

    # window: miss(c) = 1 - max_a [F(a+c) - F(a-c)]
    win = np.empty_like(C_GRID)
    for i, c in enumerate(C_GRID):
        k = int(round(c / DZ))
        if k == 0:
            win[i] = 1.0
            continue
        cov = F[2 * k:] - F[:-2 * k]
        win[i] = 1.0 - cov.max()
    return dict(var=var, mad=mad, pin90=pin(0.9), pin10=pin(0.1), win=win)


_CACHE: dict[str, dict] = {}


_EXACT_VAR = {"t3": 3.0, "t5": 5.0 / 3.0}      # grid truncation understates t variances


def _load_disk():
    import pickle, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles.pkl")
    if os.path.exists(p):
        _CACHE.update(pickle.load(open(p, "rb")))
        return
    for f in FAMILIES:
        profile(f)
    pickle.dump(_CACHE, open(p, "wb"))


def profile(fam: str) -> dict:
    if fam not in _CACHE:
        P = _profile(FAMILIES[fam])
        if fam in _EXACT_VAR:
            P["var"] = _EXACT_VAR[fam]
        _CACHE[fam] = P
    return _CACHE[fam]


def sample(fam: str, n: int, rng) -> np.ndarray:
    if fam == "normal":
        return rng.standard_normal(n)
    if fam == "laplace":
        return rng.laplace(0, 1, n)
    if fam == "gamma2":
        return rng.gamma(2.0, 1.0, n)
    if fam == "lognorm":
        return np.exp(0.6 * rng.standard_normal(n))
    if fam == "t3":
        return rng.standard_t(3, n)
    if fam == "t5":
        return rng.standard_t(5, n)
    if fam in ("mix05", "mix35"):
        p = 0.05 if fam == "mix05" else 0.35
        return rng.standard_normal(n) + 3.5 * (rng.random(n) < p)
    raise KeyError(fam)


def pointwise(fam: np.ndarray, s: np.ndarray, H=(30.0, 60.0)) -> dict:
    """Per-observation Bayes risks for arrays of family names and scales."""
    out = {k: np.empty(len(s)) for k in ["MSE", "MAE", "Q90", "Q10"] + [f"W{int(h)}" for h in H]}
    for f in np.unique(fam):
        m = fam == f
        P = profile(f)
        out["MSE"][m] = s[m] ** 2 * P["var"]
        out["MAE"][m] = s[m] * P["mad"]
        out["Q90"][m] = s[m] * P["pin90"]
        out["Q10"][m] = s[m] * P["pin10"]
        for h in H:
            out[f"W{int(h)}"][m] = np.interp(h / s[m], C_GRID, P["win"], right=0.0)
    return out


_load_disk()


if __name__ == "__main__":
    for f in FAMILIES:
        P = profile(f)
        print(f"{f:8s} var={P['var']:.4f} mad={P['mad']:.4f} pin90={P['pin90']:.4f} "
              f"win(1)={np.interp(1, C_GRID, P['win']):.4f} win(2)={np.interp(2, C_GRID, P['win']):.4f}")
