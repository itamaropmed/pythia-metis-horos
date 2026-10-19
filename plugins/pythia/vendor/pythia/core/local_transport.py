"""Shift-aware transport and sequential monitoring for local frozen-model risk certificates.

Transport (sensitivity model).  For a binary loss L in {0, 1} (e.g. W30 miss), suppose the later
population of a routed cell differs from the development population only by a density ratio
w(x, y) with  sup w / inf w <= Lambda  (Lambda = 1: no shift).  Then the later cell risk r' and the
development risk r satisfy  odds(r') in [odds(r) / Lambda, Lambda * odds(r)]  and this is sharp
(the worst case puts the large weight on L = 1).  Applying the map to a certified interval [l, u]
gives a certified interval for r' at the same confidence, *conditional on the declared Lambda*.
The tipping value Lambda* of a certified direction is the largest Lambda for which the transported
local interval and the transported global interval remain separated.

Monitoring.  ``BettingRiskMonitor`` is a test-martingale (Ville) monitor for H0: every later loss has
conditional mean <= u given the past.  It mixes fixed bets lambda_j in (0, 1/u) uniformly, so its
e-value E_t = mean_j prod_s (1 + lambda_j (L_s - u)) is a nonnegative supermartingale under H0 and
P(sup_t E_t >= 1/alpha) <= alpha: an anytime-valid alarm for harmful shift.  It assumes the later
losses are conditionally independent given the past (one case per patient, or patient-level means).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _odds(p):
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return p / (1.0 - p)


def _prob(o):
    o = np.asarray(o, float)
    return np.where(np.isinf(o), 1.0, o / (1.0 + o))


def odds_transport(lower: float, upper: float, lam: float) -> tuple[float, float]:
    """Transport a certified risk interval for a {0,1} loss under a density-ratio budget Lambda >= 1."""
    if lam < 1:
        raise ValueError("Lambda >= 1 required")
    return float(_prob(_odds(lower) / lam)), float(_prob(_odds(upper) * lam))


def transported_relation(local: tuple[float, float], glob: tuple[float, float], lam: float,
                         shift: str = "both") -> str:
    """shift="both": local and global populations may each drift by up to Lambda (worst case, sharp).
    shift="local": only this region drifts; the global reference stays at its development interval."""
    ll, lu = odds_transport(*local, lam)
    gl, gu = odds_transport(*glob, lam) if shift == "both" else glob
    if ll > gu:
        return "certified_higher_than_global"
    if lu < gl:
        return "certified_lower_than_global"
    return "local_shift_unresolved"


def tipping_lambda(local: tuple[float, float], glob: tuple[float, float], lam_max: float = 50.0,
                   shift: str = "both") -> float:
    """Largest Lambda in [1, lam_max] keeping the interval-separation direction certified
    (1.0 if the two intervals already overlap at Lambda = 1)."""
    rel = lambda lam: transported_relation(local, glob, lam, shift)
    base = rel(1.0)
    if base == "local_shift_unresolved":
        return 1.0
    if rel(lam_max) == base:
        return float(lam_max)
    lo, hi = 1.0, lam_max
    for _ in range(50):
        mid = (lo * hi) ** .5
        lo, hi = (mid, hi) if rel(mid) == base else (lo, mid)
    return float(lo)


def contrast_tipping_lambda(local: tuple[float, float], contrast: tuple[float, float],
                            glob: tuple[float, float], lam_max: float = 50.0) -> float:
    """Tipping Lambda for a direction certified by the contrast interval [cl, cu] for r_c - r_g.

    Localized-shift model: only the region drifts (odds ratio <= Lambda); the global reference r_g does
    not.  For 'higher' (cl > 0) the later relation r_c' > r_g is guaranteed iff for every feasible
    development pair (r_c, r_g) -- r_c in the region interval, r_g in the global interval and
    r_c - r_g >= cl -- the worst transported value odds^-1(odds(r_c)/Lambda) exceeds r_g.  The worst
    r_g given r_c is min(gu, r_c - cl), checked on a fine grid of feasible r_c.  'lower' is symmetric."""
    cl, cu = contrast
    lc, uc = local; gl, gu = glob
    if cl > 0:
        grid = np.linspace(max(lc, gl + cl), uc, 400)
        worst_g = np.minimum(gu, grid - cl)
        ok = lambda lam: bool(np.all(_prob(_odds(grid) / lam) > worst_g))
    elif cu < 0:
        grid = np.linspace(lc, min(uc, gu + cu), 400)
        worst_g = np.maximum(gl, grid - cu)
        ok = lambda lam: bool(np.all(_prob(_odds(grid) * lam) < worst_g))
    else:
        return 1.0
    if grid.size == 0 or not ok(1.0):
        return 1.0
    if ok(lam_max):
        return float(lam_max)
    lo, hi = 1.0, lam_max
    for _ in range(60):
        mid = (lo * hi) ** .5
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(lo)


@dataclass
class BettingRiskMonitor:
    """Anytime-valid alarm when the later risk of a routed cell exceeds its certified bound u."""
    upper: float
    alpha: float = .05
    grid: int = 20
    _logw: np.ndarray = field(init=False)
    _lams: np.ndarray = field(init=False)
    n: int = field(default=0, init=False)
    alarm_at: int | None = field(default=None, init=False)

    def __post_init__(self):
        if not 0 < self.upper < 1:
            raise ValueError("upper bound in (0, 1) required")
        self._lams = np.linspace(.05, .95, self.grid) / self.upper
        self._logw = np.zeros(self.grid)

    def update(self, loss: float) -> float:
        x = float(loss)
        self.n += 1
        self._logw += np.log1p(self._lams * (x - self.upper))
        e = float(np.exp(self._logw).mean())
        if self.alarm_at is None and e >= 1 / self.alpha:
            self.alarm_at = self.n
        return e
