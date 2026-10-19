"""Fold-error dependence, made observable.

Why this module exists
----------------------------------------------------------------------------
The Bregman identity ``R* = Risk(f) - E[D(m, f)]`` is exact. Turning it into a
number needs ``E[D(m, f)]``, and the only observable is disagreement between
predictors. Under *orthogonality* -- E[(f_i - m)(f_j - m)] = 0 -- the pairwise
statistic

    p_ij = (1/4) E[Dsym(f_i, f_j)]

equals ``(Delta_i + Delta_j) / 2`` where ``Delta_i = E[D(m, f_i)]``. Orthogonality
is the A-fold assumption, and the first M0 suite run showed it failing hard:
predictors that share approximation bias agree far more than their individual
excess risks would imply, ``p_ij`` under-states, and the bound comes out above
R* -- invalid.

The observable that fixes it
----------------------------------------------------------------------------
There is a second observable nobody was using. Because every predictor's risk
is ``R* + Delta_i``, **risk differences are excess-risk differences**:

    Delta_i - Delta_k  =  Risk_i - Risk_k  =:  d_i        (observable, exact)

So with ``Delta`` the best predictor's excess risk, ``Delta_i = Delta + d_i``, and
positing a single shared dependence ``E[(f_i - m)(f_j - m)] = lambda * sqrt(Delta_i Delta_j)``:

    p_ij  =  Delta + (d_i + d_j)/2  -  lambda * sqrt((Delta + d_i)(Delta + d_j))

That is ``B(B-1)/2`` equations in **two** unknowns, so with three or more
predictors the system is over-determined and both ``Delta`` and ``lambda`` are
identified from data alone. ``lambda`` is exactly the fold-error correlation the
ladder previously asked the user to assume.

Nothing here is fitted to a target; it is a re-parameterisation of the identity,
which is what the provenance rule requires.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import least_squares


def _residuals(theta, d_i, d_j, p, huber: float = 0.0):
    delta, lam = theta
    pred = delta + 0.5 * (d_i + d_j) - lam * np.sqrt((delta + d_i) * (delta + d_j))
    r = pred - p
    if huber > 0:
        a = np.abs(r)
        r = np.where(a <= huber, r, np.sign(r) * huber * np.sqrt(np.maximum(2 * a / huber - 1, 0)))
    return r


def solve_delta_lambda(risks: np.ndarray,
                       pair_delta: dict[tuple[int, int], float],
                       lam_max: float = 0.98) -> dict:
    """Identify (Delta, lambda) from per-predictor risks and pairwise disagreement.

    Parameters
    ----------
    risks       : (B,) out-of-fold risk of each predictor
    pair_delta  : {(i, j): p_ij} for i < j

    Returns a dict with ``delta`` (best predictor's excess risk), ``lambda``,
    ``delta_conservative`` (the most pessimistic pair at the fitted lambda),
    ``rmse`` of the fit and ``identified`` (False when there are too few pairs).
    """
    risks = np.asarray(risks, float)
    k = int(np.argmin(risks))
    d = risks - risks[k]                              # d_k = 0 by construction

    ij = sorted(pair_delta)
    d_i = np.array([d[i] for i, _ in ij])
    d_j = np.array([d[j] for _, j in ij])
    p = np.array([pair_delta[e] for e in ij])

    if len(ij) < 2:
        return {"delta": float(p.max()), "lambda": 0.0,
                "delta_conservative": float(p.max()), "rmse": float("nan"),
                "identified": False, "best": k}

    lo = np.array([0.0, 0.0])
    hi = np.array([max(10.0 * float(p.max()) + 10.0 * float(d.max()) + 1e-9, 1e-6), lam_max])
    best, best_cost = None, np.inf
    for d0 in (float(p.max()), float(p.mean()), 1e-3):
        for l0 in (0.0, 0.3, 0.6, 0.85):
            x0 = np.clip([d0, l0], lo, hi)
            try:
                r = least_squares(_residuals, x0, bounds=(lo, hi), args=(d_i, d_j, p))
            except Exception:
                continue
            if r.cost < best_cost:
                best, best_cost = r, r.cost
    if best is None:
        return {"delta": float(p.max()), "lambda": 0.0,
                "delta_conservative": float(p.max()), "rmse": float("nan"),
                "identified": False, "best": k}

    delta, lam = float(best.x[0]), float(best.x[1])

    # Most pessimistic pair at the fitted lambda: solve each pair alone for Delta
    # and keep the largest. This is the conservative reading of the same identity.
    cand = []
    for (i, jj), pij in zip(ij, p):
        di, dj = d[i], d[jj]

        def f(x):
            return x + 0.5 * (di + dj) - lam * np.sqrt((x + di) * (x + dj)) - pij

        lo_x, hi_x = 0.0, hi[0]
        if f(lo_x) > 0:                    # already above at Delta = 0
            cand.append(0.0)
            continue
        for _ in range(80):                # bisection: f is increasing in x
            mid = 0.5 * (lo_x + hi_x)
            if f(mid) < 0:
                lo_x = mid
            else:
                hi_x = mid
        cand.append(0.5 * (lo_x + hi_x))

    return {
        "delta": delta,
        "lambda": lam,
        "delta_conservative": float(max(cand)) if cand else delta,
        "rmse": float(np.sqrt(np.mean(_residuals(best.x, d_i, d_j, p) ** 2))),
        "identified": True,
        "best": k,
        "n_pairs": len(ij),
    }


def pair_delta_map(P: np.ndarray, divergence) -> dict[tuple[int, int], float]:
    out = {}
    for i, j in combinations(range(P.shape[0]), 2):
        out[(i, j)] = 0.25 * float(np.mean(divergence(P[i], P[j]) + divergence(P[j], P[i])))
    return out
