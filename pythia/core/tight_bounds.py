"""Tighter, variance-adaptive certificates for the scalar projection bound (Pythia P1, scalar outcome).

Target.  For a capped signed outcome r_i in [-cap, cap] with conditional means m_i = E[r_i | design] and a label-free
score h that sums to zero inside every table cell,
        V  =  (1/n) sum_i (m_i - mbar_cell(i))^2   >=   C^2 / H2,      C = (1/n) sum h_i m_i,  H2 = (1/n) sum h_i^2
(Cauchy-Schwarz).  The certificate is a lower confidence bound C_low on C, so V >= (C_low)_+^2 / H2.

Why a new inequality.  The previous Maurer-Pontil empirical-Bernstein bound pays 7 B log(2/a) / (3(n-1)) with
B = cap * max|h|; on testbed v2 that range term ate ~90% of the signal.  Here:

1. ``betting_lower_mean``: the predictable-plug-in empirical-Bernstein supermartingale with a constant bet lambda
   (Waudby-Smith & Ramdas 2023, PrPl-EB; Howard et al. 2021).  Proof sketch, valid for NON-constant means
   mu_t = E[x_t | past]:  Fan's inequality exp(l xi - psi_E(l) xi^2) <= 1 + l xi for xi >= -1, l in [0,1), applied
   to xi = x_t - muhat_{t-1} (both in [0,1]), gives E exp(l (x_t - mu_t) - psi_E(l)(x_t - muhat_{t-1})^2) <= 1.
   Markov on the product gives, with prob >= 1 - a,
        mean(mu) >= mean(x) - (log(1/a) + psi_E(l) sum (x_t - muhat_{t-1})^2) / (n l).
   The deviation scales like sqrt(2 var log(1/a) / n) once l is tuned; the range enters only through l < 1.
   lambda must be chosen without the certification outcomes (from FIT / training folds).
2. ``crossfit_certificate``: K-fold cross-certification.  Fold k's score is trained on the other folds and certified
   on fold k at a / K.  Since C_k <= sqrt(V_k H_k) on every fold, Cauchy-Schwarz gives the pooled certificate
        V_pooled >= (sum_k n_k C_k,low)_+^2 / (n * sum_k n_k H_k),
   so every labelled row is used both for training and for certification (no row certifies a score it trained).

Sampling unit.  With ``entity_ids`` the increments are per-entity sums (independent entities).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


def _check_alpha(alpha: float) -> None:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")


def psi_e(lam: float) -> float:
    return float(-np.log1p(-lam) - lam)


def betting_lower_mean(x: np.ndarray, alpha: float, lam: float, *, seed: int = 0) -> float:
    """Lower (1 - alpha) bound on the average conditional mean of independent x_t in [0, 1] (constant bet lam).
    The row order is a fixed seeded permutation (independent of the data)."""
    _check_alpha(alpha)
    if not (0 < lam < 1):
        raise ValueError("lam must be in (0, 1)")
    x = np.asarray(x, float)
    if x.min() < -1e-12 or x.max() > 1 + 1e-12:
        raise ValueError("x must lie in [0, 1]")
    x = np.clip(x, 0, 1)[np.random.default_rng(seed).permutation(len(x))]; n = len(x)
    muhat = (0.5 + np.concatenate([[0.0], np.cumsum(x)[:-1]])) / np.arange(1, n + 1)
    V = float(((x - muhat) ** 2).sum())
    return float(x.mean() - (np.log(1 / alpha) + psi_e(lam) * V) / (n * lam))


def lam_from_pilot(var_x: float, n: int, alpha: float, lam_max: float = 0.9) -> float:
    """Bet size that minimises the deviation for a pilot variance estimate (in the [0, 1] scale)."""
    return float(np.clip(np.sqrt(2 * np.log(1 / alpha) / (n * max(var_x, 1e-12))), 1e-4, lam_max))


def _units(entity_ids, m):
    if entity_ids is None:
        return np.arange(m), m
    _, inv = np.unique(np.asarray(entity_ids).astype(str), return_inverse=True)
    return inv, int(inv.max()) + 1


def _centred_ok(h, groups, tol=1e-8):
    if groups is None:
        return
    g = np.asarray(groups).astype(str)
    for c in np.unique(g):
        if abs(h[g == c].sum()) > tol * (1 + np.abs(h).sum()):
            raise ValueError("score h must sum to zero inside every group (centre it first)")


@dataclass
class ScalarCert:
    bound: float          # certified V (outcome units^2)
    C_lower: float
    C_hat: float
    H2: float
    plug_in: float        # C_hat^2 / H2 (not a certificate)
    n_units: int
    alpha: float
    method: str


def _increments(h, r, cap, entity_ids):
    inv, n = _units(entity_ids, len(h))
    v = np.bincount(inv, weights=h * r, minlength=n)
    B = np.bincount(inv, weights=np.abs(h) * cap, minlength=n)
    return v, B, n


def scalar_cert(h, r, cap, alpha, *, method="betting", lam=None, var_pilot=None, groups=None, entity_ids=None,
                seed=0) -> ScalarCert:
    """Certificate for V >= C^2/H2 with the chosen concentration inequality.
    method: 'betting' (PrPl-EB, constant lam: pass lam, or var_pilot = pilot variance of h r in outcome units) or
            'mp' (Maurer-Pontil empirical Bernstein, the previous default) or 'clt' (normal approximation; reference
            only, NOT a finite-sample certificate)."""
    _check_alpha(alpha)
    h = np.asarray(h, float); r = np.clip(np.asarray(r, float), -cap, cap); m = len(h)
    _centred_ok(h, groups)
    v, B, n = _increments(h, r, cap, entity_ids)
    H2 = float((h ** 2).mean()); C_hat = float(v.sum() / m)
    if H2 <= 1e-15 or n < 3:
        return ScalarCert(0.0, 0.0, C_hat, H2, 0.0, n, alpha, method)
    if method == "mp":
        l = np.log(2 / alpha)
        lo_u = v.mean() - np.sqrt(2 * v.var(ddof=1) * l / n) - 7 * (2 * B.max()) * l / (3 * (n - 1))
    elif method == "clt":
        from scipy.stats import norm
        lo_u = v.mean() - norm.ppf(1 - alpha) * v.std(ddof=1) / np.sqrt(n)
    elif method == "betting":
        Bm = float(B.max()); x = (v + Bm) / (2 * Bm)
        if lam is None:
            vp = (var_pilot if var_pilot is not None else v.var()) / (2 * Bm) ** 2   # NOTE: pass var_pilot from FIT
            lam = lam_from_pilot(vp, n, alpha)
        lo_u = betting_lower_mean(x, alpha, lam, seed=seed) * 2 * Bm - Bm
    else:
        raise ValueError(method)
    C_low = float(lo_u * n / m)
    return ScalarCert(max(C_low, 0.0) ** 2 / H2, C_low, C_hat, H2, max(C_hat, 0) ** 2 / H2, n, alpha, method)


def centre_within(h, g):
    h = np.asarray(h, float).copy(); g = np.asarray(g).astype(str)
    for c in np.unique(g):
        I = g == c; h[I] -= h[I].mean()
    return h


def tame(h, g, q):
    lo, hi = np.quantile(h, [q, 1 - q]); return centre_within(np.clip(h, lo, hi), g)


def tune_cert(h_tr_oof, r_tr, g_tr, alpha, n_eval, qs=(.0, .02, .05, .10, .20, .30), caps=(45., 60., 90., 120.),
              method="betting"):
    """Choose taming quantile, outcome cap and bet size on TRAINING rows only (out-of-fold scores).
    Simulates the certificate at the evaluation size by rescaling the deviation (uses the training rows as pilot)."""
    best = (-1.0, qs[0], caps[0], None)
    for q in qs:
        hc = tame(h_tr_oof, g_tr, q)
        for c in caps:
            rr = np.clip(r_tr, -c, c); v = hc * rr; B = np.abs(hc).max() * c
            vx = v.var() / (2 * B) ** 2; lam = lam_from_pilot(vx, n_eval, alpha)
            H2 = (hc ** 2).mean(); Cm = v.mean()
            # expected deviation at n_eval for this inequality (pilot plug-in)
            if method == "betting":
                dev = 2 * B * (np.log(1 / alpha) + psi_e(lam) * n_eval * vx) / (n_eval * lam)
            else:
                l = np.log(2 / alpha); dev = np.sqrt(2 * v.var() * l / n_eval) + 7 * 2 * B * l / (3 * n_eval)
            b = max(Cm - dev, 0) ** 2 / max(H2, 1e-15)
            if not np.isfinite(b):
                continue
            if b > best[0]:
                best = (b, q, c, lam)
    return {"q": best[1], "cap": best[2], "lam": best[3] if best[3] is not None and np.isfinite(best[3]) else .5,
            "expected_bound": max(best[0], 0.0)}


def crossfit_certificate(train_score: Callable, X, r, g, alpha, *, K=2, seed=0, method="betting",
                         entity_ids=None, tune=True, cap=60., q=.02) -> dict:
    """K-fold cross-certification on the pooled labelled rows.
    train_score(X_tr, r_tr, g_tr, X_ev, g_ev) -> raw scores for the evaluation rows (label-free for them).
    Fold k: train on the other folds (tuning q/cap/lam on their out-of-fold scores), certify on fold k at alpha/K.
    Folds are entity-disjoint when entity_ids is given."""
    _check_alpha(alpha)
    n = len(r); rng = np.random.default_rng(seed)
    if entity_ids is None:
        fold = rng.permutation(n) % K
    else:
        u, inv = np.unique(np.asarray(entity_ids).astype(str), return_inverse=True)
        fold = (rng.permutation(len(u)) % K)[inv]
    a = alpha / K; num = 0.0; den = 0.0; parts = []
    for k in range(K):
        tr, ev = fold != k, fold == k
        cfg = {"q": q, "cap": cap, "lam": None}
        if tune:
            itr = np.flatnonzero(tr); f2 = np.random.default_rng(seed + 1 + k).permutation(len(itr)) % 2
            oof = np.zeros(len(itr))
            for j in (0, 1):
                a_, b_ = itr[f2 != j], itr[f2 == j]
                oof[f2 == j] = train_score(X[a_], r[a_], g[a_], X[b_], g[b_])
            cfg = tune_cert(oof, r[itr], g[itr], a, int(ev.sum()), caps=(cap,), method=method)   # cap is part of the target: declared, not tuned
        h = tame(train_score(X[tr], r[tr], g[tr], X[ev], g[ev]), g[ev], cfg["q"])
        ce = scalar_cert(h, r[ev], cfg["cap"], a, method=method, lam=cfg["lam"], groups=g[ev],
                         entity_ids=None if entity_ids is None else np.asarray(entity_ids)[ev])
        nk = int(ev.sum()); num += nk * ce.C_lower; den += nk * ce.H2
        parts.append({"fold": k, "n": nk, "bound": ce.bound, "C_lower": ce.C_lower, "H2": ce.H2, "cap": cfg["cap"],
                      "q": cfg["q"], "lam": cfg["lam"]})
    return {"bound": max(num, 0) ** 2 / (n * den) if den > 0 else 0.0, "folds": parts, "alpha": alpha, "K": K,
            "cap": cap, "note": "pooled V >= (sum n_k C_k,low)^2 / (n sum n_k H_k); V of the outcome capped at the declared cap"}
