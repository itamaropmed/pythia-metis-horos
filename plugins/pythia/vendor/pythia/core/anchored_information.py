"""Table-anchored text information: certified, training-free information that free text adds BEYOND a tabular
record (and the Brier partial-information pieces built from it).

Objects (rows independent given (X, T); X = discrete tabular record, T = text; p_i = P(Y | X_i, T_i)):
  Gbar_XT  = mean_i (1 - |p_i|^2)                         Brier floor of table + text
  Gbar_X   = sum_c w_c (1 - |pbar_c|^2), w_c = m_c / n    Brier floor of the table alone (cells c = distinct X)
  DeltaI(T | X) = Gbar_X - Gbar_XT = sum_c w_c V_c,  V_c = mean_{i in c} |p_i - pbar_c|^2
                                                       = text information beyond the table ("anchored information")
Estimators inside each table cell (exact replicates of X, so no smoothness or shape assumption anywhere):
  Q_all = sum_c w_c [1 - sum_k n_ck (n_ck - 1)/(m_c (m_c - 1))]:  E Q_all = Gbar_XT + sum_c w_c m_c/(m_c-1) V_c
  Q_txt = disagreement of a perfect matching built INSIDE each cell from text similarity (outcome-blind):
          E Q_txt = Gbar_XT + D_txt,  D_txt >= 0
  =>  E(Q_all - Q_txt) <= sum_c w_c m_c/(m_c-1) V_c  <=  (m*/(m*-1)) DeltaI,   m* = min cell size used
  =>  DeltaI >= ((m*-1)/m*) (Q_all - Q_txt) - deviation                 [assumption-free lower bound]
Also two-sided, assumption-free: Gbar_X from Q_all (E Q_all = Gbar_X + sum_c w_c V_c/(m_c-1)), and
Gbar_XT <= Q_txt_U.
Concentration: Q_all is a weighted sum of independent within-cell U-statistics (Hoeffding: each behaves as a mean of
floor(m_c/2) independent terms); Q_txt is a mean of independent pair terms; the two share rows, so their sub-Gaussian
proxies combine as (sqrt(a) + sqrt(b))^2 (Holder)."""
from __future__ import annotations
import math
import numpy as np
from .geometry import knn_query


def _greedy_pairs(E):
    m = len(E)
    if m < 2:
        return []
    k = min(10, m - 1)
    d, idx = knn_query(E, k=k + 1)
    edges = sorted((d[i, r], i, idx[i, r]) for i in range(m) for r in range(k + 1) if idx[i, r] != i)   # ties: exclude self
    free = np.ones(m, bool); P = []
    for _, i, j in edges:
        if free[i] and free[j]:
            free[i] = free[j] = False; P.append((i, j))
    rest = list(np.flatnonzero(free))
    while len(rest) > 1:
        R = np.array(rest); D = ((E[R][:, None] - E[R][None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        a, b = np.unravel_index(np.argmin(D), D.shape); P.append((R[a], R[b])); rest = [x for t, x in enumerate(rest) if t not in (a, b)]
    flat = [v for pr in P for v in pr]
    assert len(set(flat)) == len(flat) == 2 * (m // 2), "matching is not perfect (self-pair or reuse)"
    return P


def anchored_information(cells, E, y, K, alpha=.05, min_cell=4, seed=0):
    """cells: integer table-cell id per row (distinct X).  E: text geometry (rows, dims), outcome-blind.
    Returns certified lower bound on DeltaI(T|X), two-sided bounds on Gbar_X, upper bound on Gbar_XT."""
    cells = np.asarray(cells); y = np.asarray(y); rng = np.random.default_rng(seed)
    ids = [c for c in np.unique(cells) if (cells == c).sum() >= min_cell]
    rows = np.concatenate([np.flatnonzero(cells == c) for c in ids]); n = len(rows)
    Qall, Ztxt, proxy_all, mstar = 0.0, [], 0.0, 10 ** 9
    for c in ids:
        I = np.flatnonzero(cells == c); I = I[rng.permutation(len(I))]; m = len(I); w = m / n
        cnt = np.bincount(y[I], minlength=K).astype(float)
        Qall += w * (1 - (cnt * (cnt - 1)).sum() / (m * (m - 1)))
        proxy_all += w ** 2 / (m // 2)
        mstar = min(mstar, m)
        for a, b in _greedy_pairs(E[I]):
            Ztxt.append(float(y[I[a]] != y[I[b]]))
    Ztxt = np.array(Ztxt); N = len(Ztxt); Qtxt = float(Ztxt.mean())
    L = math.log(1 / alpha)
    proxy_diff = (math.sqrt(proxy_all) + math.sqrt(1 / N)) ** 2
    dev = math.sqrt(proxy_diff * L / 2)
    dI_L = max(((mstar - 1) / mstar) * (Qall - Qtxt - dev), 0.0)
    dev_all = math.sqrt(proxy_all * L / 2); dev_txt = math.sqrt(L / (2 * N))
    return dict(dI_lower=dI_L, Q_all=Qall, Q_txt=Qtxt, GX=(max(Qall - dev_all - (1 - 1 / K) / (mstar - 1), 0.0), Qall + dev_all),
                GXT_upper=min(Qtxt + dev_txt, 1.0), n_used=int(n), cells_used=len(ids), dI_point=Qall - Qtxt)
