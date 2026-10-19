"""Smoothness-declared dominated SAA for CONTINUOUS information (no exact replicates).

Theorem (second-order smooth domination).  Rows are independent; x_j are fixed covariates; the
conditional law of Y given x has Bayes action a*(x) (median / q-quantile) that is L-Lipschitz
(declared), and its density is bounded by f_max/s(x) near the action (declared curvature
kappa_j = c_curv * f_max / s_j, c_curv = 2 for MAE, 1 for pinball).  Partition rows outcome-blind
into blocks b with frozen centres c_b.  For a cap tau,

  S  = (1/n) sum_b min_a sum_{j in b} min(l(Y_j - a), tau)                 (observable, exact)
  W_j = min(l(Y_j - a*(c_b)), tau)  in [0, tau], independent             (unobservable)

Then S <= mean_j W_j pathwise, and since l(Y - a) has curvature <= kappa and first-order slope
<= c_q (c_q = 1 for MAE, max(q,1-q) for pinball),

  E W_j <= R_j + pen_j,   pen_j = min( c_q |a*(x_j) - a*(c_b)| , kappa_j/2 (a*(x_j) - a*(c_b))^2 )
                                 <= min( c_q L d_j , kappa_j/2 L^2 d_j^2 ),   d_j = ||x_j - c_b||.

Hence R := (1/n) sum_j R_j >= Bentkus^{-1}(S; n, tau, alpha) - (1/n) sum_j pen_j.

The penalty is computable from features alone.  Unlike pair/kNN witnesses, (i) concentration
runs over all n rows (domination), not over n/k block scores, and (ii) the drift penalty is
second order, so large blocks (small in-block optimism) remain affordable.
"""
from __future__ import annotations
import math
import numpy as np
from .dominated_saa import truncated_cell_sum, bentkus_lower_from_statistic


def kd_blocks(X, k, seed=0):
    X = np.asarray(X, float).reshape(len(X), -1)
    n = (len(X) // k) * k
    idx = np.random.default_rng(seed).permutation(len(X))[:n]
    out, stack = [], [idx]
    while stack:
        I = stack.pop()
        if len(I) == k:
            out.append(I); continue
        j = int(np.argmax(X[I].max(0) - X[I].min(0)))
        I = I[np.argsort(X[I, j], kind="stable")]
        cut = k * ((len(I) // k) // 2)
        stack += [I[:cut], I[cut:]]
    return np.array(out)


def _greedy_match(C):
    n = len(C)
    D = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(n, 1)
    order = np.argsort(D[iu], kind="stable")
    used = np.zeros(n, bool); P = []
    for e in order:
        i, j = iu[0][e], iu[1][e]
        if not used[i] and not used[j]:
            used[i] = used[j] = True; P.append((i, j))
            if len(P) == n // 2:
                break
    return P


def matching_blocks(X, k):
    """Outcome-blind recursive greedy nearest-neighbour matching: blocks of size k = 2^j with small
    diameter in any dimension (pairs of points, then pairs of pair-centroids, ...)."""
    X = np.asarray(X, float).reshape(len(X), -1)
    if k & (k - 1):
        raise ValueError("k must be a power of two")
    groups = [[i] for i in range(len(X))]
    while len(groups[0]) < k:
        C = np.array([X[g].mean(0) for g in groups])
        groups = [groups[i] + groups[j] for i, j in _greedy_match(C)]
    return np.array(groups)


def balanced_blocks(X, k, init=None, iters=6, seed=0):
    """Outcome-blind balanced k-means (exact size k) via optimal assignment (Hungarian on
    k-replicated centroids) = balanced optimal-transport partition minimising sum ||x - c_b||^2.
    Initialised from matching (if k power of 2) or kd blocks; monotone non-increasing objective."""
    from scipy.optimize import linear_sum_assignment
    X = np.asarray(X, float).reshape(len(X), -1)
    B = init if init is not None else (matching_blocks(X, k) if not (k & (k - 1)) else kd_blocks(X, k, seed))
    idx = B.ravel(); m = len(B)
    Xs = X[idx]
    lab = np.repeat(np.arange(m), k)
    best = None
    for _ in range(iters):
        C = np.array([Xs[lab == j].mean(0) for j in range(m)])
        cost = ((Xs[:, None, :] - C[None, :, :]) ** 2).sum(-1)          # (n_used, m)
        rows, cols = linear_sum_assignment(np.repeat(cost, k, axis=1))
        new = cols // k
        obj = cost[rows, new].sum()
        if best is not None and obj >= best - 1e-12:
            break
        best, lab = obj, new
    return np.array([idx[lab == j] for j in range(m)])


def smooth_penalty(X, blocks, L, kappa_rows, c_q, order="best", metric="l2", local_moduli=None, kappa_fn=None):
    """metric='l2': |a*(x)-a*(x')| <= L ||x-x'||_2.  metric='l1' with X pre-scaled by per-coordinate
    moduli L_i and L=1: |a*(x)-a*(x')| <= sum_i L_i |x_i - x'_i| (declared anisotropic/sparse smoothness)."""
    X = np.asarray(X, float).reshape(len(X), -1)
    Xb = X[blocks]
    c = Xb.mean(1, keepdims=True)
    if local_moduli is not None:
        # declared LOCAL per-coordinate moduli L_{b,i} = sup over the block's bounding box of |d a*/d x_i|;
        # the segment x_j -> c_b lies in that box, so |a*(x_j) - a*(c_b)| <= sum_i L_{b,i} |x_ji - c_bi|.
        Lb = np.asarray(local_moduli(blocks), float)               # (m, dim)
        d = (np.abs(Xb - c) * Lb[:, None, :]).sum(-1)
        L = 1.0
    else:
        d = np.sqrt(((Xb - c) ** 2).sum(-1)) if metric == "l2" else np.abs(Xb - c).sum(-1)
    first = c_q * L * d
    if kappa_fn is not None:
        # declared LOCAL curvature: sup of the conditional density only over [a* - delta, a* + delta],
        # delta = L d (the action drift actually possible for that row); kappa_fn(rows, delta) -> curvature
        kap_loc = kappa_fn(blocks.ravel(), (L * d).ravel()).reshape(d.shape)
        second = kap_loc / 2 * (L * d) ** 2
    else:
        second = kappa_rows[blocks] / 2 * (L * d) ** 2
    pen = {"first": first, "second": second, "best": np.minimum(first, second)}[order]
    return float(pen.sum())


def smooth_dominated_lower_bound(X, y, *, loss, L, kappa_rows, block_sizes, caps, alpha=.05,
                                 quantile=None, seed=0, order="best",
                                 geometries=("kd", "matching"),
                                 concentrations=("domination_bentkus", "block_dkelly"), metric="l2",
                                 local_moduli=None, kappa_fn=None):
    """Portfolio over outcome-blind geometries x block sizes x caps x concentration routes, all
    Bonferroni-charged.  Two concentration routes share the same witness and the same penalty:
      domination_bentkus : S <= mean_j W_j over all n rows; Bentkus with range tau (n units);
      block_dkelly       : independent block scores in [0,tau] (m = n/k units); dKelly (variance-adaptive).
    Both are valid lower endpoints for R + mean penalty; the penalty is subtracted afterwards."""
    from .truncated_saa import dkelly_lower_mean
    y = np.asarray(y, float); n = len(y)
    q = .5 if loss == "mae" else float(quantile)
    c_q = 1.0 if loss == "mae" else max(q, 1 - q)
    kap = np.asarray(kappa_rows, float)
    n_cand = len(geometries) * len(block_sizes) * len(caps) * len(concentrations)
    a_each = alpha / n_cand
    best = (0.0, None)
    for geo in geometries:
        for k in block_sizes:
            if geo == "matching" and k & (k - 1):
                continue
            if geo == "matching":
                B = matching_blocks(X, k)
            elif geo == "balanced":
                B = balanced_blocks(X, k, seed=seed)
            else:
                B = kd_blocks(X, k, seed)
            pen = smooth_penalty(X, B, L, kap, c_q, order, metric, local_moduli, kappa_fn) / n
            m = len(B)
            for tau in caps:
                sc = np.array([truncated_cell_sum(y[b], loss=loss, kappa=tau, quantile=quantile) for b in B])
                stat = sc.sum() / n
                for conc in concentrations:
                    if conc == "domination_bentkus":
                        lcb = bentkus_lower_from_statistic(stat, n, tau, a_each)
                    else:
                        lcb = dkelly_lower_mean(sc / k, cap=tau, alpha=a_each) * m * k / n
                    val = max(0.0, lcb - pen)
                    if val > best[0]:
                        best = (val, {"geometry": geo, "k": k, "cap": tau, "concentration": conc,
                                      "lcb": lcb, "pen": pen, "stat": stat})
    return best


def twonn_dimension(X, discard=0.1):
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017): MLE from ratios mu = r2/r1 of the two nearest
    neighbour distances, discarding the largest `discard` fraction.  Planning tool only (outcome-blind);
    under a declared anisotropic metric (X scaled by L_i) it gives the *effective* dimension of the penalty."""
    from scipy.spatial import cKDTree
    X = np.asarray(X, float).reshape(len(X), -1)
    d, _ = cKDTree(X).query(X, k=3)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    mu = mu[: int(len(mu) * (1 - discard))]
    return float(len(mu) / np.sum(np.log(mu)))


def planned_penalty_curve(X, block_sizes, L, kappa, geometry="kd"):
    """Outcome-blind feasibility plan: mean second-order penalty per row for each block size, plus the
    TwoNN dimension.  Compare with a prior guess of the floor to decide whether certification is worthwhile."""
    X = np.asarray(X, float).reshape(len(X), -1)
    out = {"twonn_dimension": twonn_dimension(X), "curve": []}
    for k in block_sizes:
        B = kd_blocks(X, k) if geometry == "kd" else matching_blocks(X, k)
        Xb = X[B]; c = Xb.mean(1, keepdims=True)
        d2 = ((Xb - c) ** 2).sum(-1).mean()
        out["curve"].append({"k": k, "mean_d2": float(d2), "second_order_penalty": float(kappa / 2 * L ** 2 * d2)})
    return out
