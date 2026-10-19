"""Numerical CDF-integral gold for the frozen synthetic clipped laws.

Used only for validation, never in Peras statistics or confidence radii.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

import gold

_CACHE = {}


def risks(fam: str, mu: float, scale: float, clip=(0., 300.)) -> dict:
    key = (str(fam), float(mu), float(scale), tuple(clip))
    if key in _CACHE:
        return _CACHE[key]
    lo, hi = clip
    x = np.linspace(lo, hi, 12_001)
    F = gold.FAMILIES[fam]((x - mu) / scale)
    left = cumulative_trapezoid(F, x, initial=0.)
    right = cumulative_trapezoid(1 - F, x, initial=0.)
    mean = lo + right[-1]
    second = lo ** 2 + np.trapezoid(2 * x * (1 - F), x)
    out = {"MSE": float(max(0., second - mean ** 2))}
    for q in (.1, .5, .9):
        a = float(np.interp(q, F, x))
        risk = ((1 - q) * np.interp(a, x, left)
                + q * (right[-1] - np.interp(a, x, right)))
        out[f"Q{int(q * 100)}"] = float(risk)
    out["MAE"] = 2 * out["Q50"]
    for h in (15., 30., 60.):
        if 2 * h >= hi - lo:
            out[f"W{int(h)}"] = 0.
            continue
        steps = int(round(2 * h / (x[1] - x[0])))
        covered = F[steps:] - F[:-steps]
        # Clipping creates endpoint atoms; include them in closed windows.
        covered[0] = F[steps]
        covered[-1] = 1 - F[-steps - 1]
        out[f"W{int(h)}"] = float(max(0., 1 - covered.max()))
    _CACHE[key] = out
    return out


def pointwise_risks(fam, mu, scale, clip=(0., 300.), grid_size=3_001):
    """Vectorized numeric clipped gold for heterogeneous location-scale laws."""
    fam = np.asarray(fam).astype(str).ravel()
    mu = np.asarray(mu, float).ravel()
    scale = np.asarray(scale, float).ravel()
    if (len(fam) != len(mu) or len(mu) != len(scale) or not len(mu)
            or not np.isfinite(mu).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0) or grid_size < 501):
        raise ValueError("aligned finite laws, positive scales, and adequate grid required")
    lo, hi = map(float, clip)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("finite ordered clip required")
    x = np.linspace(lo, hi, int(grid_size))
    out = {key: np.empty(len(mu)) for key in ("MAE", "MSE", "Q10", "Q50", "Q90")}
    for family in np.unique(fam):
        mask = fam == family
        f = gold.FAMILIES[family]((x[None, :] - mu[mask, None]) / scale[mask, None])
        left = cumulative_trapezoid(f, x, axis=1, initial=0.)
        right = cumulative_trapezoid(1 - f, x, axis=1, initial=0.)
        mean = lo + right[:, -1]
        second = lo ** 2 + np.trapezoid(2 * x[None, :] * (1 - f), x, axis=1)
        out["MSE"][mask] = np.maximum(0., second - mean ** 2)
        for q in (.1, .5, .9):
            actions = np.array([np.interp(q, row, x) for row in f])
            lower = np.array([np.interp(a, x, row) for a, row in zip(actions, left)])
            upper = np.array([np.interp(a, x, row) for a, row in zip(actions, right)])
            out[f"Q{int(q * 100)}"][mask] = ((1 - q) * lower
                                                       + q * (right[:, -1] - upper))
        out["MAE"][mask] = 2 * out["Q50"][mask]
    return out
