"""Instrumented information: certified accounting of what *readers* (LLMs, extractors, embeddings) take from text.

A reader is any map rho: (X, T) -> R that is fixed without the evaluation labels (a prompt + model + decoding
config, an embedding, a rule).  For quadratic information (Brier geometry) and design-conditional truth
p_i = P(Y | X_i, T_i):

  V(G)      = (1/m) sum_i |p_i - pbar_{G(i)}|^2        within-group spread of the true conditional law
  NLI(T|X)  = V(cells of X)                            information the text holds beyond the table (in-sample)

Results implemented here (proofs in docs/theory/INSTRUMENTED_INFORMATION_20261019.md):

  P1 projection bound.  For ANY label-free score matrix H that sums to zero inside every group of G,
       V(G) >= (C_lower)_+^2 / mean|H|^2,  C = mean_i H[i, y_i],
     with C_lower a one-sided lower confidence bound for C (labels independent given the design).
  P2 reader portfolio.  Readers 1..J declared up front: max_j bound_j at alpha/J is valid.  Choosing the
     reader after seeing labels costs only the union bound -- "LLM hacking" cannot inflate a lower bound
     beyond log J in the deviation.
  P3 information ledger.  For a reader A with label-free bins a(i) (refining the table cells):
       V(X) = V(X, a) + B(a),   B(a) = (1/m) sum_i |pbar_{X,a} - pbar_X|^2   (Pythagoras, exact in-sample)
     B(a) >= P1 with H constant on (X, a) groups; V(X, a) >= P1 with any second reader centred on (X, a).
     Both hold simultaneously at alpha/2 each, so  NLI(T|X) >= L_B(a) + L_V(X,a)  (certified additive ledger).
  P4 regression test.  For readers old/new: L_V(X, a_new) computed with the OLD reader's scores > 0 certifies
     that the text holds information inside the new reader's bins that the old reader finds: the upgrade
     lost information (or never had it).
  P5 selective reading.  A policy that runs an expensive reader only on cases chosen by a label-free rule is
     itself a reader, so P1-P4 certify the information retained per unit cost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .concentration import heterogeneous_chernoff_lower_sum


def _groups(*keys) -> np.ndarray:
    """Integer group id from one or more label arrays (outcome-free)."""
    arr = np.stack([np.asarray(k).astype(str) for k in keys], 1)
    _, g = np.unique(arr, axis=0, return_inverse=True)
    return g.ravel()


def loo_centre_groups(Q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """h_i = q_i - mean_{j != i, same group} q_j ; zero for singleton groups. Sums to zero in every group."""
    H = np.zeros_like(Q, dtype=float)
    for c in np.unique(g):
        I = np.flatnonzero(g == c); m = len(I)
        if m >= 2:
            S = Q[I].sum(0); H[I] = Q[I] - (S[None] - Q[I]) / (m - 1)
    return H


def between_scores(Q: np.ndarray, coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
    """Label-free score constant on fine groups and exactly centred inside coarse groups (P3, between part)."""
    H = np.zeros_like(Q, dtype=float)
    for c in np.unique(coarse):
        I = np.flatnonzero(coarse == c)
        cell_mean = Q[I].mean(0)
        for f in np.unique(fine[I]):
            J = I[fine[I] == f]
            H[J] = Q[J].mean(0) - cell_mean
    return H


@dataclass
class ProjectionBound:
    bound: float
    C: float
    C_lower: float
    H2: float
    rho: float
    n: int
    alpha: float
    method: str
    extra: dict = field(default_factory=dict)


def _check_alpha(alpha: float) -> None:
    if not (0 < float(alpha) < 1):
        raise ValueError("alpha must be in (0, 1)")


def _units(entity_ids, m: int):
    """Independent sampling units: rows (default) or clusters given by entity_ids (e.g. patients)."""
    if entity_ids is None:
        return np.arange(m), m, 1, "independent_rows_assumption"
    e = np.asarray(entity_ids).astype(str).reshape(-1)
    if len(e) != m:
        raise ValueError("entity_ids must align with rows")
    _, inv, counts = np.unique(e, return_inverse=True, return_counts=True)
    return inv, int(inv.max()) + 1, int(counts.max()), "independent_clusters_assumption"


def projection_bound(H: np.ndarray, y: np.ndarray, alpha: float, method: str = "chernoff", *,
                     groups, entity_ids=None, tol: float = 1e-8) -> ProjectionBound:
    """P1: lower confidence bound on V(G) for a label-free H centred within the groups of G.

    Conditional on the design, x_i = H[i, y_i] - min_k H[i, k] lie in [0, w_i].  With entity_ids the
    independent units are clusters: X_e = sum_{i in e} x_i in [0, sum_{i in e} w_i] (arbitrary dependence
    inside a cluster).  The mean of sum_i x_i / m is sum_i <p_i, h_i>/m - mean(min_k H); Cauchy-Schwarz with
    sum_{i in group} h_i = 0 gives C <= sqrt(V(G) * mean|H|^2).  Uncentred scores are refused (they could
    otherwise certify the group means rather than within-group information)."""
    _check_alpha(alpha)
    H = np.asarray(H, float); y = np.asarray(y, int); m = len(y)
    g = np.asarray(groups).reshape(-1)
    if len(g) != m or H.shape[0] != m:
        raise ValueError("H, y and groups must align")
    scale = 1.0 + np.abs(H).sum()
    for c in np.unique(g.astype(str)):
        if np.abs(H[g.astype(str) == c].sum(0)).max() > tol * scale:
            raise ValueError("score matrix H must sum to zero inside every group of G (centre it first)")
    inv, n_units, max_rows, assumption = _units(entity_ids, m)
    z = H[np.arange(m), y]; lo = H.min(1); w = H.max(1) - lo
    C = float(z.mean()); H2 = float((H ** 2).sum(1).mean())
    extra = {"n_independent_clusters": n_units, "max_rows_per_cluster": max_rows,
             "sampling_unit_assumption": assumption}
    if H2 <= 1e-12 or w.max() <= 1e-15:
        return ProjectionBound(0.0, C, 0.0, H2, 0.0, m, alpha, method, extra)
    X = np.bincount(inv, weights=z - lo, minlength=n_units); W = np.bincount(inv, weights=w, minlength=n_units)
    if method == "hoeffding":
        C_lower = C - math.sqrt((W ** 2).sum() * math.log(1 / alpha) / 2) / m
    elif method == "chernoff":
        C_lower = float(lo.mean() + heterogeneous_chernoff_lower_sum(np.clip(X, 0, W), W, alpha) / m)
    else:
        raise ValueError(method)
    return ProjectionBound(max(C_lower, 0.0) ** 2 / H2, C, float(C_lower), H2, C / math.sqrt(H2), m, alpha, method,
                           extra)


def _fold_weights(fold_ids, m):
    f = np.asarray(fold_ids).reshape(-1)
    if len(f) != m:
        raise ValueError("fold_ids must align with rows")
    folds = list(np.unique(f))
    return f, folds, {k: float((f == k).mean()) for k in folds}


def reader_portfolio(cells, readers: dict, y, alpha: float, method: str = "chernoff", *,
                     fold_ids=None, entity_ids=None) -> dict:
    """P2: readers = {name: held-out probability matrix (m x K)}.  Every bound at alpha / J; report the max.
    fold_ids: pooled out-of-fold scores are certified inside each fold block (scores of a fold depend only on
    the other folds' outcomes), alpha / (J * F) per block, and combined by fold size."""
    _check_alpha(alpha)
    y = np.asarray(y, int); J = len(readers)
    if fold_ids is None:
        g = _groups(cells)
        res = {k: projection_bound(loo_centre_groups(np.asarray(Q, float), g), y, alpha / J, method,
                                   groups=g, entity_ids=entity_ids) for k, Q in readers.items()}
        best = max(res, key=lambda k: res[k].bound)
        return {"bound": res[best].bound, "selected": best, "alpha_each": alpha / J, "foldwise": False,
                "per_reader": {k: r.bound for k, r in res.items()}}
    f, folds, wts = _fold_weights(fold_ids, len(y)); cells = np.asarray(cells)
    ent = None if entity_ids is None else np.asarray(entity_ids)
    per = {k: 0.0 for k in readers}; fold_bounds = {}
    for fo in folds:
        ix = f == fo; gi = _groups(cells[ix]); fb = {}
        for k, Q in readers.items():
            Qi = np.asarray(Q, float)[ix]
            fb[k] = projection_bound(loo_centre_groups(Qi, gi), y[ix], alpha / (J * len(folds)), method, groups=gi,
                                     entity_ids=None if ent is None else ent[ix]).bound
            per[k] += wts[fo] * fb[k]
        fold_bounds[str(fo)] = fb
    best = max(per, key=per.get)
    return {"bound": per[best], "selected": best, "alpha_each": alpha / (J * len(folds)), "foldwise": True,
            "per_reader": per, "fold_bounds": fold_bounds}


def _ledger_once(cells, bins_a, Q_a, Q_b, y, a, method, entity_ids):
    gx = _groups(cells); gxa = _groups(cells, bins_a)
    Lb = projection_bound(between_scores(np.asarray(Q_a, float), gx, gxa), y, a, method, groups=gx, entity_ids=entity_ids)
    Lw = projection_bound(loo_centre_groups(np.asarray(Q_b, float), gxa), y, a, method, groups=gxa, entity_ids=entity_ids)
    return Lb, Lw


def information_ledger(cells, bins_a, Q_a, Q_b, y, alpha: float, method: str = "chernoff", *,
                       fold_ids=None, entity_ids=None) -> dict:
    """P3: NLI(T|X) >= L_between(a) + L_within(X, a); each part at alpha/2 (alpha / (2F) per fold block).

    bins_a: reader A's label-free discretisation; Q_a: A's held-out probabilities (for the between part);
    Q_b: any second reader's held-out probabilities (for the information A's bins miss)."""
    _check_alpha(alpha)
    y = np.asarray(y, int)
    if fold_ids is None:
        Lb, Lw = _ledger_once(cells, bins_a, Q_a, Q_b, y, alpha / 2, method, entity_ids)
        return {"between_a": Lb.bound, "beyond_a": Lw.bound, "total": Lb.bound + Lw.bound,
                "rho_between": Lb.rho, "rho_beyond": Lw.rho, "foldwise": False}
    f, folds, wts = _fold_weights(fold_ids, len(y)); cells = np.asarray(cells); bins_a = np.asarray(bins_a)
    ent = None if entity_ids is None else np.asarray(entity_ids)
    out = {"between_a": 0.0, "beyond_a": 0.0, "total": 0.0, "foldwise": True, "fold_totals": []}
    for fo in folds:
        ix = f == fo
        Lb, Lw = _ledger_once(cells[ix], bins_a[ix], np.asarray(Q_a)[ix], np.asarray(Q_b)[ix], y[ix],
                              alpha / (2 * len(folds)), method, None if ent is None else ent[ix])
        out["between_a"] += wts[fo] * Lb.bound; out["beyond_a"] += wts[fo] * Lw.bound
        out["fold_totals"].append(Lb.bound + Lw.bound)
    out["total"] = out["between_a"] + out["beyond_a"]
    return out


def regression_test(cells, bins_new, Q_old, y, alpha: float, method: str = "chernoff", *,
                    fold_ids=None, entity_ids=None) -> dict:
    """P4: certified information inside the NEW reader's bins that the OLD reader recovers."""
    _check_alpha(alpha)
    y = np.asarray(y, int)
    if fold_ids is None:
        g = _groups(cells, bins_new)
        r = projection_bound(loo_centre_groups(np.asarray(Q_old, float), g), y, alpha, method, groups=g,
                             entity_ids=entity_ids)
        return {"lost_information_lower": r.bound, "regression_certified": r.bound > 0, "rho": r.rho,
                "foldwise": False}
    f, folds, wts = _fold_weights(fold_ids, len(y)); cells = np.asarray(cells); bins_new = np.asarray(bins_new)
    ent = None if entity_ids is None else np.asarray(entity_ids); total = 0.0; fb = []
    for fo in folds:
        ix = f == fo; g = _groups(cells[ix], bins_new[ix])
        b = projection_bound(loo_centre_groups(np.asarray(Q_old, float)[ix], g), y[ix], alpha / len(folds), method,
                             groups=g, entity_ids=None if ent is None else ent[ix]).bound
        fb.append(b); total += wts[fo] * b
    return {"lost_information_lower": total, "regression_certified": total > 0, "foldwise": True, "fold_bounds": fb}


def true_spread(P: np.ndarray, *keys) -> float:
    """Simulation helper: V(G) for a known design-conditional probability matrix P."""
    g = _groups(*keys); tot = 0.0
    for c in np.unique(g):
        I = np.flatnonzero(g == c)
        tot += ((P[I] - P[I].mean(0)) ** 2).sum()
    return tot / len(P)



def _eb_lower_mean(x: np.ndarray, b: float, alpha: float) -> float:
    """Maurer-Pontil empirical-Bernstein one-sided lower bound for the mean of independent x_i in [0, b].
    (Valid for independent, non-identically distributed bounded variables as a bound on the average mean.)"""
    n = len(x); v = x.var(ddof=1) if n > 1 else b * b / 4; l = np.log(2 / alpha)
    return float(x.mean() - np.sqrt(2 * v * l / n) - 7 * b * l / (3 * (n - 1)))


def projection_bound_scalar(h: np.ndarray, z: np.ndarray, cap: float, alpha: float,
                            method: str = "best") -> ProjectionBound:
    """P1 for a bounded scalar outcome z in [0, cap] (e.g. a capped absolute error): lower confidence bound on
    V = (1/m) sum (m_i - mbar_G)^2, m_i = E[z_i | design], for a label-free h that sums to zero in every group.
    Conditional on the design x_i = h_i z_i - min(0, h_i cap) are independent in [0, |h_i| cap]."""
    h = np.asarray(h, float); z = np.asarray(z, float); m = len(z)
    lo = np.minimum(0, h * cap); w = np.abs(h) * cap; x = h * z - lo
    C = float((h * z).mean()); H2 = float((h ** 2).mean())
    if H2 <= 1e-15 or w.max() <= 1e-15:
        return ProjectionBound(0.0, C, 0.0, H2, 0.0, m, alpha, "chernoff_scalar")
    a = alpha / 2 if method == "best" else alpha            # 'best' = both endpoints at alpha/2 (union bound)
    ch = float(lo.mean() + heterogeneous_chernoff_lower_sum(np.clip(x, 0, w), w, a) / m)
    eb = float(lo.mean() + _eb_lower_mean(np.clip(x, 0, w), float(w.max()), a))
    C_lower = {"chernoff": ch, "eb": eb, "best": max(ch, eb)}[method]
    return ProjectionBound(max(C_lower, 0) ** 2 / H2, C, C_lower, H2, C / np.sqrt(H2 * max(z.var(), 1e-15)), m, alpha,
                           f"scalar_{method}")


def loo_centre_scalar(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    return loo_centre_groups(np.asarray(q, float)[:, None], g)[:, 0]



def tame_scores(raw: np.ndarray, g: np.ndarray, q: float = .02) -> np.ndarray:
    """Label-free taming of a score before centring: winsorise at the q / 1-q quantiles of the evaluation scores,
    then leave-one-out centre within groups.  Range-based concentration pays for max|h|, so a few extreme scores
    can erase the certificate; winsorising changes only the score (never the outcomes) and keeps P1 valid."""
    raw = np.asarray(raw, float)
    if raw.ndim == 1:
        lo, hi = np.quantile(raw, [q, 1 - q]); return loo_centre_scalar(np.clip(raw, lo, hi), g)
    lo, hi = np.quantile(raw, [q, 1 - q], axis=0); return loo_centre_groups(np.clip(raw, lo, hi), g)



def eb_scalar_bound(h: np.ndarray, r: np.ndarray, cap: float, alpha: float, *, groups=None, entity_ids=None,
                    tol: float = 1e-8) -> float:
    """P1 for a capped signed outcome r in [-cap, cap] and a centred label-free score h, with the Maurer-Pontil
    empirical-Bernstein inequality applied to v_i = h_i r_i directly (no per-row shift, so the empirical variance
    is not inflated by heterogeneous offsets).  Returns the certified within-group variance of E[r | design]."""
    _check_alpha(alpha)
    h = np.asarray(h, float); r = np.clip(np.asarray(r, float), -cap, cap); m = len(h)
    if groups is not None:
        g = np.asarray(groups).astype(str)
        for c in np.unique(g):
            if abs(h[g == c].sum()) > tol * (1 + np.abs(h).sum()):
                raise ValueError("score h must sum to zero inside every group (centre it first)")
    inv, n, _, _ = _units(entity_ids, m)
    v = np.bincount(inv, weights=h * r, minlength=n); Bu = np.bincount(inv, weights=np.abs(h) * cap, minlength=n)
    H2 = float((h ** 2).mean())
    if H2 <= 1e-15 or n < 3:
        return 0.0
    l = np.log(2 / alpha)
    # mean over units of v_e (each in [-B_e, B_e]); EB with the largest unit range; rescale to a per-row mean
    lower_units = v.mean() - np.sqrt(2 * v.var(ddof=1) * l / n) - 7 * (2 * Bu.max()) * l / (3 * (n - 1))
    C_lower = lower_units * n / m
    return float(max(C_lower, 0.0) ** 2 / H2)


def tune_on_fit(h_fit_oof: np.ndarray, r_fit: np.ndarray, g_fit: np.ndarray, alpha: float,
                qs=(.02, .05, .10), caps=(60., 120.)) -> tuple:
    """Choose the taming quantile and the outcome cap on FIT rows only (out-of-fold scores), never on CAL."""
    best = (-1.0, qs[0], caps[0])
    for q in qs:
        lo, hi = np.quantile(h_fit_oof, [q, 1 - q])
        hc = loo_centre_scalar(np.clip(h_fit_oof, lo, hi), g_fit)
        for c in caps:
            b = eb_scalar_bound(hc, r_fit, c, alpha)
            if b > best[0]:
                best = (b, q, c)
    return best[1], best[2]
