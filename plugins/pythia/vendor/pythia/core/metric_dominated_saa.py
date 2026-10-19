"""Metric-agnostic smooth domination: Bayes-floor certificates for information living in ANY metric space
(e.g. embedding-free text geometry: character n-gram Jaccard, compression distance NCD).

Theorem.  Rows independent; information x_j in a metric space (M, D); the Bayes action a*(x) satisfies the
declared modulus |a*(x) - a*(x')| <= L D(x, x') (for 0-1 loss: TV(p(.|x), p(.|x')) <= L D(x,x')).  Partition
rows outcome-blind into blocks with a MEDOID m_b (a data point, so no vector space is needed).  With
W_j = min(l(Y_j - a*(x_{m_b})), tau) (0-1: 1{Y_j != argmax p(.|x_{m_b})}),
   S = (1/n) sum_b min_a sum_{j in b} min(l(Y_j - a), tau) <= mean_j W_j,
   E W_j <= R_j + pen_j,  pen_j = min(c_q L D_j, kappa_j/2 L^2 D_j^2)  (0-1: 2 L D_j),  D_j = D(x_j, x_{m_b}).
Hence R >= LCB(S) - mean pen.  Only pairwise distances to medoids are needed."""
from __future__ import annotations
import numpy as np
from .dominated_saa import truncated_cell_sum, bentkus_lower_from_statistic
from .truncated_saa import dkelly_lower_mean


def _greedy_capacity_assign(cost, k):
    """Assign every row to a column with capacity k, greedily by increasing cost (outcome-blind)."""
    n, m = cost.shape
    order = np.argsort(cost, axis=None, kind="stable")
    lab = np.full(n, -1); cap = np.full(m, k); left = min(n, m * k)
    for e in order:
        i, b = divmod(int(e), m)
        if lab[i] < 0 and cap[b] > 0:
            lab[i] = b; cap[b] -= 1; left -= 1
            if left == 0:
                break
    return lab


def balanced_kmedoids(D, k, iters=8, seed=0):
    """Outcome-blind balanced k-medoids on a distance matrix: blocks of exact size k (leftover rows dropped,
    contributing a safe zero).  Assignment by Hungarian on k-replicated medoid columns."""
    from scipy.optimize import linear_sum_assignment
    n = len(D); m = n // k
    rng = np.random.default_rng(seed)
    # farthest-point initialisation
    med = [int(rng.integers(n))]
    for _ in range(m - 1):
        med.append(int(np.argmax(D[:, med].min(1))))
    med = np.array(med)
    best = None
    for _ in range(iters):
        cost = D[:, med]                                   # (n, m)
        lab = _greedy_capacity_assign(cost, k)
        ok = lab >= 0
        obj = cost[np.flatnonzero(ok), lab[ok]].sum()
        newmed = med.copy()
        for b in range(m):
            mem = np.flatnonzero(lab == b)
            newmed[b] = mem[np.argmin(D[np.ix_(mem, mem)].sum(1))]
        if best is not None and obj >= best - 1e-12:
            break
        best, med = obj, newmed
    lab = _greedy_capacity_assign(D[:, med], k)
    blocks = [np.flatnonzero(lab == b) for b in range(m)]
    return blocks, med


def metric_dominated_lower_bound(D, y, *, loss, L, block_sizes, caps=(1.0,), alpha=.05, quantile=None,
                                 kappa_rows=None, oracle_action=None, seed=0):
    """Portfolio over block sizes x caps x {domination-Bentkus, block-dKelly}.  If `oracle_action` (array a*(x))
    is given, the penalty uses the exact |a*(x_j) - a*(x_m)| instead of L*D (simulation ceiling)."""
    y = np.asarray(y, float); n = len(y)
    q = .5 if loss == "mae" else (quantile if loss == "pinball" else None)
    c_q = 1.0 if loss == "mae" else (max(q, 1 - q) if loss == "pinball" else None)
    caps = (1.0,) if loss == "zero_one" else tuple(caps)
    a_each = alpha / (len(block_sizes) * len(caps) * 2)
    best = (0.0, None)
    for k in block_sizes:
        blocks, med = balanced_kmedoids(D, k, seed=seed)
        dist = np.concatenate([D[b, med[i]] for i, b in enumerate(blocks)])
        rows = np.concatenate(blocks)
        if oracle_action is not None:
            drift = np.concatenate([np.abs(oracle_action[b] - oracle_action[med[i]]) for i, b in enumerate(blocks)])
        else:
            drift = L * dist
        if loss == "zero_one":
            pen = (2 * drift).sum() / n
        else:
            kap = np.asarray(kappa_rows, float)[rows]
            pen = np.minimum(c_q * drift, kap / 2 * drift ** 2).sum() / n
        for tau in caps:
            sc = np.array([truncated_cell_sum(y[b], loss=loss, kappa=tau, quantile=quantile) for b in blocks])
            stat = sc.sum() / n
            for conc, lcb in (("domination_bentkus", bentkus_lower_from_statistic(stat, n, tau, a_each)),
                              ("block_dkelly", dkelly_lower_mean(sc / k, cap=tau, alpha=a_each) * len(blocks) * k / n)):
                val = max(0.0, lcb - pen)
                if val > best[0]:
                    best = (val, {"k": k, "cap": tau, "concentration": conc, "pen": pen, "stat": stat})
    return best
