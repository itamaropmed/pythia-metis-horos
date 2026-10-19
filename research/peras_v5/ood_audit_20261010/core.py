"""Loss-generic tuple machinery.

For a tuple of k outcomes (rows of a matrix T), the *in-tuple ERM risk* is
    r(T) = min_a (1/k) sum_j loss(T_j - a)
and the *leave-one-out risk* of a point e given k neighbour values N is
    loss(e - a_hat(N)),  a_hat(N) = argmin_a sum_j loss(N_j - a).

Both are exact for MSE, MAE, pinball and window-miss loss (the optimum of a
piecewise-linear / step loss is attained at a data point, so we search over the
tuple values themselves).
"""
from __future__ import annotations
import numpy as np

LOSSES = ["MSE", "MAE", "Q90", "W30", "W60"]


def _pin(u, q):
    return np.maximum(q * u, (q - 1) * u)


def pointloss(L, u):
    if L == "MSE":
        return u ** 2
    if L == "MAE":
        return np.abs(u)
    if L == "Q90":
        return _pin(u, 0.9)
    if L.startswith("W"):
        return (np.abs(u) > float(L[1:])).astype(float)
    raise KeyError(L)


def erm_risk(L, T):
    """Row-wise min_a mean loss(T - a).  T: (n, k)."""
    if L == "MSE":
        return T.var(axis=1)
    if L == "MAE":
        med = np.median(T, axis=1, keepdims=True)
        return np.abs(T - med).mean(axis=1)
    if L == "Q90":
        # candidates are the tuple values
        R = _pin(T[:, :, None] - T[:, None, :], 0.9).mean(axis=1)   # (n, cand)
        return R.min(axis=1)
    if L.startswith("W"):
        h = float(L[1:])
        S = np.sort(T, axis=1)
        cover = (S[:, None, :] <= S[:, :, None] + 2 * h) & (S[:, None, :] >= S[:, :, None])
        return 1.0 - cover.sum(axis=2).max(axis=1) / T.shape[1]
    raise KeyError(L)


def erm_action(L, N):
    """Row-wise argmin_a sum loss(N - a)."""
    if L == "MSE":
        return N.mean(axis=1)
    if L == "MAE":
        return np.median(N, axis=1)
    if L == "Q90":
        R = _pin(N[:, :, None] - N[:, None, :], 0.9).sum(axis=1)
        return N[np.arange(len(N)), R.argmin(axis=1)]
    if L.startswith("W"):
        h = float(L[1:])
        S = np.sort(N, axis=1)
        cover = (S[:, None, :] <= S[:, :, None] + 2 * h) & (S[:, None, :] >= S[:, :, None])
        cnt = cover.sum(axis=2)
        j = cnt.argmax(axis=1)
        rows = np.arange(len(N))
        lo = S[rows, j]
        hi_idx = j + cnt[rows, j] - 1
        hi = S[rows, hi_idx]
        return 0.5 * (lo + hi)          # centre of the covered points
    raise KeyError(L)


def loo_risk(L, e, N):
    return pointloss(L, e - erm_action(L, N))
