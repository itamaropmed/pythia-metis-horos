"""Peras v5: training-free, loss-agnostic estimation of attainable risk.

Pipeline (no predictive model is fitted or stored):
 1. Representation (outcome-blind): numeric columns standardised; binary/categorical columns and
    LLM-read note concepts enter with a large weight (near-exact matching).
 2. Near-replicate tuples: each case + its k-1 nearest neighbours.
 3. Local-linear detrending (closed form, leave-tuple-out): a ridge-regularised weighted
    linear fit on the case's m-neighbourhood *excluding the tuple* removes first-order location
    heterogeneity inside the tuple.  Spacings of the detrended tuple ~ spacings of i.i.d. noise.
 4. Per-loss statistics on detrended tuples: in-tuple ERM (optimistic) and leave-one-out
    (pessimistic) risks; estimator = optimism-cancelling combination, extrapolated to zero
    tuple spread (two-scale Richardson in (1/k, radius^2)).
"""
from __future__ import annotations
import sys, json, numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import core
from core import erm_risk, loo_risk

LOSSES = ["MSE", "MAE", "Q90", "W30", "W60"]
RIDGE = {"relative": None}      # None: absolute ridge (v5, v5.4); float a: penalty_j = a * sum_m Xf_j^2 (scale-free, v5.5)


def _penalty(A, p, ridge):
    if RIDGE["relative"] is None:
        return ridge * np.diag(np.r_[0.0, np.ones(p)])[None]
    sc = np.einsum("nmp,nmp->np", A[..., 1:], A[..., 1:])
    sc = np.maximum(sc, RIDGE.get("floor", 0.0) * A.shape[1]) + 1e-9
    P = np.zeros((A.shape[0], p + 1, p + 1)); idx = np.arange(1, p + 1)
    P[:, idx, idx] = RIDGE["relative"] * sc
    return P


def represent(d, w_disc=3.0, use_llm=True):
    tab = d["tab"]
    cols = []
    for j in range(tab.shape[1]):
        v = tab[:, j]; u = np.unique(v)
        if len(u) <= 2:                        # binary (e.g. one-hot codes): near-exact match
            cols.append(w_disc * (v - v.mean()) / (v.std() + 1e-12) / np.sqrt(2))
        else:
            cols.append((v - v.mean()) / (v.std() + 1e-12))
    Z = np.column_stack(cols) if cols else np.zeros((len(tab), 0))
    if use_llm:
        C = d["llm"]
        Z = np.column_stack([Z] + [w_disc * (C[:, j] - C[:, j].mean()) / (C[:, j].std() + 1e-12) for j in range(C.shape[1])])
    return Z


def neighbours(Z, m):
    dist, nb = NearestNeighbors(n_neighbors=m + 1).fit(Z).kneighbors(Z)
    # drop self (first column usually self; guard for duplicates)
    out_nb = np.empty((len(Z), m), int); out_d = np.empty((len(Z), m))
    for i in range(len(Z)):
        msk = nb[i] != i
        out_nb[i] = nb[i][msk][:m]; out_d[i] = dist[i][msk][:m]
    return out_d, out_nb


def detrended_tuples(Z, y, nb, k, m, ridge=1.0, detrend=True):
    """T[i] = detrended outcomes of the tuple {i} + nb[i,:k-1] (relative to case i's location)."""
    n, p = Z.shape
    idx = np.column_stack([np.arange(n), nb[:, :k - 1]])          # (n,k)
    T = y[idx].astype(float)
    if not detrend:
        return T, idx
    fit_idx = nb[:, k - 1:m]                                      # neighbourhood minus tuple
    Xf = Z[fit_idx] - Z[:, None, :]                               # (n, m-k+1, p) centred at i
    yf = y[fit_idx]
    A = np.concatenate([np.ones(Xf.shape[:2] + (1,)), Xf], axis=2)  # intercept + slopes
    AtA = np.einsum("nmp,nmq->npq", A, A) + _penalty(A, p, ridge)
    Aty = np.einsum("nmp,nm->np", A, yf)
    beta = np.linalg.solve(AtA, Aty[..., None])[..., 0][:, 1:]    # slopes (n,p)
    dz = Z[idx] - Z[:, None, :]                                   # (n,k,p)
    T = T - np.einsum("nkp,np->nk", dz, beta)
    return T, idx


def tuple_stats(T, L):
    """per-case in-tuple ERM risk (using the whole tuple) and LOO risk of case 0 vs the rest."""
    r = erm_risk(L, T)
    rt = loo_risk(L, T[:, 0], T[:, 1:])
    return r, rt


QUAD = {"on": False, "cols": None}


def _expand(Zc):
    """add squares and pairwise products of the numeric coordinates (local-quadratic detrending)."""
    cols = QUAD["cols"]; S = Zc[..., cols]; q = S.shape[-1]
    feats = [S[..., a] * S[..., b] for a in range(q) for b in range(a, q)]
    return np.concatenate([Zc, np.stack(feats, -1)], -1)


def detrend_full(Z, y, nb, k, m, ridge=1.0):
    """Detrended tuples + per-case slope-noise covariance factors (for SIMEX) + local scale."""
    n, p = Z.shape
    idx = np.column_stack([np.arange(n), nb[:, :k - 1]])
    fit_idx = nb[:, k - 1:m]
    Xf = Z[fit_idx] - Z[:, None, :]; yf = y[fit_idx]
    if QUAD["on"]:
        Xf = _expand(Xf); p = Xf.shape[-1]
    A = np.concatenate([np.ones(Xf.shape[:2] + (1,)), Xf], axis=2)
    AtA = np.einsum("nmp,nmq->npq", A, A) + _penalty(A, p, ridge)
    Aty = np.einsum("nmp,nm->np", A, yf)
    coef = np.linalg.solve(AtA, Aty[..., None])[..., 0]
    res = yf - np.einsum("nmp,np->nm", A, coef)
    invf = np.linalg.inv(AtA)
    trH = np.einsum("nmp,npq,nmq->n", A, invf, A)                  # effective parameters (ridge)
    dof = np.maximum(fit_idx.shape[1] - trH, 3.0)
    s2 = (res ** 2).sum(1) / dof
    inv = invf[:, 1:, 1:]                                         # slope block
    dz = Z[idx] - Z[:, None, :]
    if QUAD["on"]:
        dz = _expand(dz)
    T = y[idx] - np.einsum("nkp,np->nk", dz, coef[:, 1:])
    # covariance of the induced trend across tuple members: dz Cov(beta) dz^T
    Cb = s2[:, None, None] * inv
    Ctr = (dz, Cb)
    rho2 = (dz[..., :Z.shape[1]] ** 2).sum(2)[:, 1:].mean(1)                         # mean squared radius
    scale = np.sqrt(np.median(res ** 2, axis=1) / 0.4549)          # robust local sd
    return T, Ctr, rho2, scale


def simex_draw(T, Ctr, lam, rng):
    if lam == 0: return T
    dz, Cb = Ctr; n, k, p = dz.shape
    Lc = np.linalg.cholesky(Cb + 1e-9 * np.eye(p)[None])
    xi = np.einsum("npq,nq->np", Lc, rng.standard_normal((n, p)))
    return T + np.sqrt(lam) * np.einsum("nkp,np->nk", dz, xi)


def dcor(x, y):
    def cen(v):
        D = np.abs(v[:, None] - v[None, :]); return D - D.mean(0) - D.mean(1)[:, None] + D.mean()
    A, B = cen(x), cen(y)
    v = (A * B).mean(); return np.sqrt(max(v, 0) / np.sqrt((A * A).mean() * (B * B).mean()))


def screen(d, rng, max_keep_all=10, n_sub=800, B=50):
    """Model-free relevance screening of numeric columns (only when there are many):
    keep columns whose distance correlation with y exceeds the 99th pct of a permutation null."""
    tab = d["tab"]; num = [j for j in range(tab.shape[1]) if len(np.unique(tab[:, j])) > 2]
    if len(num) <= max_keep_all:
        return list(range(tab.shape[1]))
    sub = rng.choice(len(tab), min(n_sub, len(tab)), replace=False); y = d["y"][sub]
    keep = [j for j in range(tab.shape[1]) if j not in num]
    null = [dcor(tab[sub, num[0]], rng.permutation(y)) for _ in range(B)]
    thr = np.quantile(null, 0.99)
    for j in num:
        if dcor(tab[sub, j], y) > thr: keep.append(j)
    return sorted(keep)


WIN = ("W30", "W60")


def estimate(d, ks=(4, 6, 8, 12), lams=(0.0, 0.5, 1.0, 2.0), m=60, nsim=2, seed=0, window="calib", use_llm=True,
             screen_cols=True, losses=LOSSES):
    """Peras-v5 point estimate of the attainable (Bayes) risk for every loss."""
    rng = np.random.default_rng(seed)
    dd = dict(d)
    if screen_cols:
        cols = screen_forward(d, rng, tol=0.01); dd["tab"] = d["tab"][:, cols]
    Z = represent(dd, use_llm=use_llm); y = d["y"]
    _, nb = neighbours(Z, m)
    rows = {L: [] for L in losses}
    for k in ks:
        T, Ctr, rho2, sc = detrend_full(Z, y, nb, k, m)
        for lam in lams:
            for b in range(nsim if lam > 0 else 1):
                Tl = simex_draw(T, Ctr, lam, rng)
                for L in losses:
                    r, rt = tuple_stats(Tl, L)
                    if L in WIN and window == "calib":
                        from methods import calibrate_window_weight
                        h = float(L[1:]); cg = np.geomspace(0.05, 40, 40)
                        z = (Tl[:, 0] - np.median(Tl[:, 1:], 1)) / np.maximum(sc, 1e-6)
                        wg = calibrate_window_weight(z, k, cg, B=2000, seed=k)
                        wi = np.interp(np.log(h / np.maximum(sc, 1e-6)), np.log(cg), wg)
                        c = float(np.mean(wi * r + (1 - wi) * rt))
                    else:
                        c = 0.5 * (r.mean() + rt.mean())
                    rows[L].append((k, lam, c, float(rho2.mean()), float((rho2 ** 2).mean()), float(r.mean()), float(rt.mean())))
    out = {}
    for L in losses:
        a = np.array(rows[L]); X = np.column_stack([np.ones(len(a)), 1 + a[:, 1]])
        out[L] = float(max(np.linalg.lstsq(X, a[:, 2], rcond=None)[0][0], 0.0))
    out["RMSE"] = float(np.sqrt(out["MSE"])) if "MSE" in out else None
    out["_rows"] = {L: rows[L] for L in losses}
    return out


def screen_forward(d, rng, max_keep_all=10, tol=0.02, use_llm=True):
    """Model-free forward selection of numeric columns by the Rice/1-NN local-variance statistic
    (Gamma-test-style feature selection; no predictor is fitted).  Only used when there are many."""
    tab = d["tab"]; num = [j for j in range(tab.shape[1]) if len(np.unique(tab[:, j])) > 2]
    if len(num) <= max_keep_all:
        return list(range(tab.shape[1]))
    y = d["y"]; keep = [j for j in range(tab.shape[1]) if j not in num]
    sel = []
    def rice(cols):
        dd = dict(d); dd["tab"] = tab[:, sorted(keep + cols)] if cols or keep else tab[:, :0]
        Z = represent(dd, use_llm=use_llm)
        if Z.shape[1] == 0: return np.var(y)
        _, nb = neighbours(Z, 40)
        T, _ = detrended_tuples(Z, y, nb, 4, 40)
        r, rt = tuple_stats(T, "MSE")
        return 0.5 * (r.mean() + rt.mean())
    cur = rice(sel)
    while True:
        cand = [(rice(sel + [j]), j) for j in num if j not in sel]
        if not cand: break
        best, j = min(cand)
        if best < cur * (1 - tol):
            sel.append(j); cur = best
        else:
            break
    return sorted(keep + sel)


# ------------------------------------------------------------------ frozen configuration (dev seeds 100-103)
CFG = {"MSE": dict(w=0.58, ks=(4, 6, 8)), "MAE": dict(w=0.6, ks=(4, 6, 8, 12)), "Q90": dict(w=0.5, ks=(4, 6, 8)),
       "W30": dict(w=0.5, ks=(4, 6, 8)), "W60": dict(w=0.55, ks=(4, 6, 8, 12))}
WIT = json.load(open(Path(__file__).with_name("witness_ws.json")))["w"]


def witness_vals(T, L):
    """designed lower-witness statistic per tuple (pinball losses), in loss units."""
    S = np.diff(np.sort(T, axis=1), axis=1); k = T.shape[1]
    if L == "MAE": return 2 * S @ np.array(WIT[f"0.5_{k}_lo"])
    if L == "Q90": return S @ np.array(WIT[f"0.9_{k}_lo"])
    raise KeyError(L)


def peras5(d, seed=0, lams=(0.0, 0.5, 1.0, 2.0), m=60, nsim=2, z=1.645):
    """Frozen Peras-v5 estimator AND lower confidence bound (training-free)."""
    rng = np.random.default_rng(seed)
    cols = screen_forward(d, rng, tol=0.01); dd = dict(d); dd["tab"] = d["tab"][:, cols]
    Z = represent(dd); y = d["y"]; n = len(y)
    _, nb = neighbours(Z, m)
    rows = {L: [] for L in LOSSES}; lcbrows = {}
    for k in (3, 4, 6, 8, 12, 16):
        T, Ctr, rho2, sc = detrend_full(Z, y, nb, k, m)
        for lam in lams:
            for b in range(nsim if lam > 0 else 1):
                Tl = simex_draw(T, Ctr, lam, rng)
                for L in LOSSES:
                    r, rt = tuple_stats(Tl, L)
                    rows[L].append((k, lam, r.mean(), rt.mean()))
                    if True:
                        lcbrows.setdefault((L, k, "saa"), []).append((lam, r.mean(), r.std(ddof=1)))
                        if k in (8, 12) and L in ("MAE", "Q90"):
                            wv = witness_vals(Tl, L); lcbrows.setdefault((L, k, "wit"), []).append((lam, wv.mean(), wv.std(ddof=1)))
    est, lcb = {}, {}
    for L in LOSSES:
        a = np.array(rows[L]); a = a[np.isin(a[:, 0], CFG[L]["ks"])]; w = CFG[L]["w"]
        c = w * a[:, 2] + (1 - w) * a[:, 3]
        est[L] = float(max(np.linalg.lstsq(np.column_stack([np.ones(len(a)), 1 + a[:, 1]]), c, rcond=None)[0][0], 0))
    for key, v in lcbrows.items():
        L, k, kind = key; a = np.array(v)
        ext = np.linalg.lstsq(np.column_stack([np.ones(len(a)), 1 + a[:, 0]]), a[:, 1], rcond=None)[0][0]
        sd = a[a[:, 0] == 0, 2][0]
        lcb[f"{L}|{k}|{kind}"] = float(ext - z * sd * np.sqrt(k) / np.sqrt(n))
        lcb[f"{L}|{k}|{kind}|raw"] = float(a[a[:, 0] == 0, 1][0] - z * sd * np.sqrt(k) / np.sqrt(n))
    est["RMSE"] = float(np.sqrt(est["MSE"]))
    return est, lcb, cols

LCB_PICK = {"MSE": "MSE|12|saa", "MAE": "MAE|4|saa|raw", "Q90": "Q90|12|wit", "W30": "W30|12|saa|raw", "W60": "W60|16|saa|raw"}


# ------------------------------------------------------------------ v5.2 development: EGOP metric
def local_slopes(Z, y, nb, m, ridge=1.0):
    """leave-self-out local linear slopes of y and of |detrended residual| at each case."""
    n, p = Z.shape
    Xf = Z[nb[:, :m]] - Z[:, None, :]; yf = y[nb[:, :m]]
    A = np.concatenate([np.ones(Xf.shape[:2] + (1,)), Xf], axis=2)
    AtA = np.einsum("nmp,nmq->npq", A, A) + ridge * np.diag(np.r_[0.0, np.ones(p)])[None]
    coef = np.linalg.solve(AtA, np.einsum("nmp,nm->np", A, yf)[..., None])[..., 0]
    res = np.abs(yf - np.einsum("nmp,np->nm", A, coef))
    coef2 = np.linalg.solve(AtA, np.einsum("nmp,nm->np", A, res)[..., None])[..., 0]
    return coef[:, 1:], coef2[:, 1:], res.mean()


def egop_transform(Z, y, numeric_mask, m=60, a=0.2, scale_w=1.0):
    """Expected-gradient-outer-product metric on the numeric coordinates (location + dispersion
    gradients); binary / LLM coordinates keep their fixed large weight."""
    _, nb = neighbours(Z, m)
    b, g, s = local_slopes(Z, y, nb, m)
    num = np.flatnonzero(numeric_mask)
    if len(num) == 0: return Z
    B = b[:, num]; Gm = g[:, num]
    M = B.T @ B / len(B) / (np.var(y) + 1e-12) + scale_w * Gm.T @ Gm / len(Gm) / (s ** 2 + 1e-12)
    M = (1 - a) * M / (np.trace(M) / len(num) + 1e-12) + a * np.eye(len(num))
    w, V = np.linalg.eigh(M); R = V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T
    Z2 = Z.copy(); Z2[:, num] = Z[:, num] @ R
    return Z2


def peras52(d, seed=0, lams=(0.0, 0.5, 1.0, 2.0), m=60, nsim=2, z=1.645, egop=0.2, scale_w=1.0, text_svd=None, cfg=None):
    """v5.2: v5 + EGOP metric (+ optional unsupervised TF-IDF coordinates when no LLM reader)."""
    rng = np.random.default_rng(seed)
    cols = screen_forward(d, rng, tol=0.01); dd = dict(d); dd["tab"] = d["tab"][:, cols]
    if text_svd is not None:
        dd["tab"] = np.column_stack([dd["tab"], text_svd])
    Z = represent(dd); y = d["y"]; n = len(y)
    numeric = np.r_[[len(np.unique(dd["tab"][:, j])) > 2 for j in range(dd["tab"].shape[1])],
                    np.zeros(Z.shape[1] - dd["tab"].shape[1], bool)]
    if egop is not None:
        Z = egop_transform(Z, y, numeric, m=m, a=egop, scale_w=scale_w)
    _, nb = neighbours(Z, m)
    cfg = cfg or CFG
    rows = {L: [] for L in LOSSES}; lcbrows = {}
    for k in (3, 4, 6, 8, 12, 16):
        T, Ctr, rho2, sc = detrend_full(Z, y, nb, k, m)
        for lam in lams:
            for b in range(nsim if lam > 0 else 1):
                Tl = simex_draw(T, Ctr, lam, rng)
                for L in LOSSES:
                    r, rt = tuple_stats(Tl, L)
                    rows[L].append((k, lam, r.mean(), rt.mean()))
                    lcbrows.setdefault((L, k, "saa"), []).append((lam, r.mean(), r.std(ddof=1)))
                    if k in (8, 12) and L in ("MAE", "Q90"):
                        wv = witness_vals(Tl, L); lcbrows.setdefault((L, k, "wit"), []).append((lam, wv.mean(), wv.std(ddof=1)))
    est, lcb = {}, {}
    for L in LOSSES:
        a_ = np.array(rows[L]); a_ = a_[np.isin(a_[:, 0], cfg[L]["ks"])]; w = cfg[L]["w"]
        c = w * a_[:, 2] + (1 - w) * a_[:, 3]
        est[L] = float(max(np.linalg.lstsq(np.column_stack([np.ones(len(a_)), 1 + a_[:, 1]]), c, rcond=None)[0][0], 0))
    for key, v in lcbrows.items():
        L, k, kind = key; a_ = np.array(v)
        ext = np.linalg.lstsq(np.column_stack([np.ones(len(a_)), 1 + a_[:, 0]]), a_[:, 1], rcond=None)[0][0]
        sd = a_[a_[:, 0] == 0, 2][0]
        lcb[f"{L}|{k}|{kind}"] = float(ext - z * sd * np.sqrt(k) / np.sqrt(n))
        lcb[f"{L}|{k}|{kind}|raw"] = float(a_[a_[:, 0] == 0, 1][0] - z * sd * np.sqrt(k) / np.sqrt(n))
    est["RMSE"] = float(np.sqrt(est["MSE"]))
    est["_rows"] = rows
    return est, lcb, cols


# ------------------------------------------------------------------ v5.3 development: fixed-k dilation Richardson
def detrend_sets(Z, y, nb, k, D, m, ridge=1.0):
    """tuple = case + every D-th neighbour (ranks D, 2D, ..); slope fit on the m-neighbourhood minus the tuple."""
    n, p = Z.shape
    tsel = np.arange(1, k) * D - 1
    idx = np.column_stack([np.arange(n), nb[:, tsel]])
    mask = np.ones(m, bool); mask[tsel] = False
    fit_idx = nb[:, :m][:, mask]
    Xf = Z[fit_idx] - Z[:, None, :]; yf = y[fit_idx]
    A = np.concatenate([np.ones(Xf.shape[:2] + (1,)), Xf], axis=2)
    AtA = np.einsum("nmp,nmq->npq", A, A) + ridge * np.diag(np.r_[0.0, np.ones(p)])[None]
    coef = np.linalg.solve(AtA, np.einsum("nmp,nm->np", A, yf)[..., None])[..., 0]
    res = yf - np.einsum("nmp,np->nm", A, coef)
    invf = np.linalg.inv(AtA)
    trH = np.einsum("nmp,npq,nmq->n", A, invf, A)
    s2 = (res ** 2).sum(1) / np.maximum(fit_idx.shape[1] - trH, 3.0)
    dz = Z[idx] - Z[:, None, :]
    T = y[idx] - np.einsum("nkp,np->nk", dz, coef[:, 1:])
    r2 = (dz ** 2).sum(2)[:, 1:]
    return T, (dz, s2[:, None, None] * invf[:, 1:, 1:]), float(r2.mean()), float((r2 ** 2).mean())


def peras53(d, seed=0, lams=(0.0, 0.5, 1.0, 2.0), m=80, nsim=2, Ds=(1, 2, 3), cfg=None, curv="rho4", per_k=False):
    rng = np.random.default_rng(seed)
    cols = screen_forward(d, rng, tol=0.01); dd = dict(d); dd["tab"] = d["tab"][:, cols]
    Z = represent(dd); y = d["y"]
    _, nb = neighbours(Z, m)
    cfg = cfg or CFG
    rows = {L: [] for L in LOSSES}
    ks = sorted({k for L in LOSSES for k in cfg[L]["ks"]})
    for k in ks:
        for D in Ds:
            if (k - 1) * D + 10 > m: continue
            T, Ctr, r2, r4 = detrend_sets(Z, y, nb, k, D, m)
            for lam in lams:
                for b in range(nsim if lam > 0 else 1):
                    Tl = simex_draw(T, Ctr, lam, rng)
                    for L in LOSSES:
                        if k not in cfg[L]["ks"]: continue
                        r, rt = tuple_stats(Tl, L)
                        rows[L].append((k, D, lam, r2, r4, r.mean(), rt.mean()))
    est = {}
    for L in LOSSES:
        a = np.array(rows[L]); w = cfg[L]["w"]; c = w * a[:, 5] + (1 - w) * a[:, 6]
        reg = a[:, 4] if curv == "rho4" else a[:, 3]
        if per_k:
            vals = []
            for k in np.unique(a[:, 0]):
                mk = a[:, 0] == k
                vals.append(np.linalg.lstsq(np.column_stack([np.ones(mk.sum()), 1 + a[mk, 2], reg[mk]]), c[mk], rcond=None)[0][0])
            est[L] = float(max(np.mean(vals), 0))
        else:
            X = np.column_stack([np.ones(len(a)), 1 + a[:, 2], reg] + [(a[:, 0] == k).astype(float) for k in np.unique(a[:, 0])[1:]])
            est[L] = float(max(np.linalg.lstsq(X, c, rcond=None)[0][0], 0))
    est["RMSE"] = float(np.sqrt(est["MSE"])); est["_rows"] = rows
    return est


# ------------------------------------------------------------------ v5.4: v5 + adaptive (shrunk) curvature correction
def curvature_correction(d, seed=0, cfg=None, m=80, D2=3, lams=(0.0, 0.5, 1.0, 2.0), nsim=2, ks=(4, 6, 8)):
    """Delta = SIMEX-extrapolated combination at dilation D2 minus at D=1 (same k): curvature inflation grows
    with rho^4 while the 1/k optimism does not change.  Returns per-loss (correction, t-stat)."""
    rng = np.random.default_rng(seed + 991)
    cols = screen_forward(d, np.random.default_rng(seed), tol=0.01); dd = dict(d); dd["tab"] = d["tab"][:, cols]
    Z = represent(dd); y = d["y"]; n = len(y)
    _, nb = neighbours(Z, m)
    cfg = cfg or CFG
    acc = {L: [] for L in LOSSES}
    for k in ks:
        stats = {}
        for D in (1, D2):
            T, Ctr, r2, r4 = detrend_sets(Z, y, nb, k, D, m)
            per = {}
            for lam in lams:
                for b in range(nsim if lam > 0 else 1):
                    Tl = simex_draw(T, Ctr, lam, rng)
                    for L in LOSSES:
                        w = cfg[L]["w"]; r, rt = tuple_stats(Tl, L); ci = w * r + (1 - w) * rt
                        per.setdefault(L, []).append((lam, ci.mean(), ci if lam == 0 else None))
            stats[D] = (per, r4)
        for L in LOSSES:
            ext = {}
            for D in (1, D2):
                a = np.array([(lam, mu) for lam, mu, _ in stats[D][0][L]])
                ext[D] = np.linalg.lstsq(np.column_stack([np.ones(len(a)), 1 + a[:, 0]]), a[:, 1], rcond=None)[0][0]
            c1 = [ci for lam, _, ci in stats[1][0][L] if ci is not None][0]; c2 = [ci for lam, _, ci in stats[D2][0][L] if ci is not None][0]
            delta = ext[D2] - ext[1]; se = np.std(c2 - c1, ddof=1) * np.sqrt(k) / np.sqrt(n)
            r41, r42 = stats[1][1], stats[D2][1]
            acc[L].append((delta * r41 / max(r42 - r41, 1e-12), delta / max(se, 1e-12)))
    out = {}
    for L in LOSSES:
        a = np.array(acc[L]); corr = a[:, 0].mean(); t = a[:, 1].mean()
        shrink = max(0.0, 1 - 1 / t ** 2) if (t > 0 and corr > 0) else 0.0
        out[L] = (float(corr), float(t), float(shrink))
    return out


def peras54(d, seed=0, **kw):
    est, lcb, cols = peras5(d, seed=seed)
    cc = curvature_correction(d, seed=seed, **kw)
    e2 = dict(est)
    for L in LOSSES:
        corr, t, sh = cc[L]
        e2[L] = float(max(est[L] - sh * corr, 0.0))
    e2["RMSE"] = float(np.sqrt(e2["MSE"])); e2["_curv"] = cc
    return e2, lcb, cols


# ------------------------------------------------------------------ v5.4 (frozen after design-balanced dev selection)
CFG54 = {"MSE": dict(w=0.55, ks=(4, 6, 8, 12), spread=None), "MAE": dict(w=0.40, ks=(6, 8, 12), spread="rho4"),
         "Q90": dict(w=0.55, ks=(6, 8, 12), spread=None), "W30": dict(w=0.50, ks=(6, 8, 12), spread=None),
         "W60": dict(w=0.65, ks=(8, 12), spread=None)}


def peras54(d, seed=0, lams=(0.0, 0.5, 1.0, 2.0), m=60, nsim=2, z=1.645, text_svd=None, use_llm=True):
    """Peras-v5.4 estimator (+ the v5 lower bound).  Optional unsupervised text coordinates (TF-IDF SVD)
    replace the LLM reader when it is unavailable."""
    rng = np.random.default_rng(seed)
    dsel = dict(d)
    if text_svd is not None:
        dsel["tab"] = np.column_stack([d["tab"], text_svd])
    if not use_llm:
        dsel["llm"] = d["llm"][:, :0]
    cols = screen_forward(dsel, rng, tol=0.01); dd = dict(dsel); dd["tab"] = dsel["tab"][:, cols]
    Z = represent(dd); y = d["y"]; n = len(y)
    _, nb = neighbours(Z, m)
    rows = {L: [] for L in LOSSES}; lcbrows = {}
    for k in (3, 4, 6, 8, 12, 16):
        T, Ctr, rho2, sc = detrend_full(Z, y, nb, k, m)
        r4 = float((rho2 ** 2).mean())
        for lam in lams:
            for b in range(nsim if lam > 0 else 1):
                Tl = simex_draw(T, Ctr, lam, rng)
                for L in LOSSES:
                    r, rt = tuple_stats(Tl, L)
                    rows[L].append((k, lam, r.mean(), rt.mean(), r4))
                    lcbrows.setdefault((L, k, "saa"), []).append((lam, r.mean(), r.std(ddof=1)))
                    if k in (8, 12) and L in ("MAE", "Q90"):
                        wv = witness_vals(Tl, L); lcbrows.setdefault((L, k, "wit"), []).append((lam, wv.mean(), wv.std(ddof=1)))
    est = {}
    for L in LOSSES:
        c_ = CFG54[L]; a = np.array(rows[L]); a = a[np.isin(a[:, 0], c_["ks"])]
        if RIDGE.get("guard"):
            keep = []
            for k in np.unique(a[:, 0]):
                ak = a[a[:, 0] == k]; cc = c_["w"] * ak[:, 2] + (1 - c_["w"]) * ak[:, 3]
                g0 = cc[ak[:, 1] == 0].mean(); g1 = cc[ak[:, 1] == 1].mean()
                if g0 > 0 and (g1 / g0 - 1) < RIDGE["guard"]: keep.append(k)
            if len(keep) >= (2 if c_["spread"] == "rho4" else 1):
                a = a[np.isin(a[:, 0], keep)]
        c = c_["w"] * a[:, 2] + (1 - c_["w"]) * a[:, 3]
        cols_ = [np.ones(len(a)), 1 + a[:, 1]] + ([a[:, 4]] if c_["spread"] == "rho4" else [])
        est[L] = float(max(np.linalg.lstsq(np.column_stack(cols_), c, rcond=None)[0][0], 0))
    lcb = {}
    for key, v in lcbrows.items():
        L, k, kind = key; a = np.array(v)
        ext = np.linalg.lstsq(np.column_stack([np.ones(len(a)), 1 + a[:, 0]]), a[:, 1], rcond=None)[0][0]
        sd = a[a[:, 0] == 0, 2][0]
        lcb[f"{L}|{k}|{kind}"] = float(ext - z * sd * np.sqrt(k) / np.sqrt(n))
        lcb[f"{L}|{k}|{kind}|raw"] = float(a[a[:, 0] == 0, 1][0] - z * sd * np.sqrt(k) / np.sqrt(n))
    est["RMSE"] = float(np.sqrt(est["MSE"]))
    return est, {L: lcb[LCB_PICK[L]] for L in LOSSES}, cols


def text_svd(notes, k=16, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(notes)
    return TruncatedSVD(k, random_state=seed).fit_transform(X)


def peras55(d, seed=0, **kw):
    """Peras-v5.5 = v5.4 with a scale-free (relative) ridge in the local detrending fits (leak-robust)."""
    old = RIDGE["relative"]; RIDGE["relative"] = 0.01
    try:
        return peras54(d, seed=seed, **kw)
    finally:
        RIDGE["relative"] = old



def peras56(d, seed=0, **kw):
    """Peras-v5.6 = v5.4 with a scale-free ridge floored at 1% of a unit-variance coordinate (stable + leak-robust)."""
    old = (RIDGE["relative"], RIDGE.get("floor", 0.0)); RIDGE["relative"] = 0.01; RIDGE["floor"] = 0.01
    try:
        return peras54(d, seed=seed, **kw)
    finally:
        RIDGE["relative"], RIDGE["floor"] = old



def peras57(d, seed=0, **kw):
    """Peras-v5.7 = v5.6 + SIMEX guard: tuple sizes whose induced-trend contamination is not small
    (g(lambda=1)/g(0) - 1 >= 0.3) are dropped (SIMEX is a small-contamination expansion)."""
    old = dict(RIDGE); RIDGE.update(relative=0.01, floor=0.01, guard=0.3)
    try:
        return peras54(d, seed=seed, **kw)
    finally:
        RIDGE.clear(); RIDGE.update(old)
