"""Full-sample SAA certificates by fixed-action domination (no blocks, no outcome support).

Theorem (fixed-action domination).  Let independent entity clusters ``g = 1..G`` carry
cases ``i`` with frozen, outcome-blind finite cells ``c(i)``; dependence inside a cluster
is arbitrary.  For a nonnegative loss ``l``, cap ``kappa > 0`` and any FIXED actions
``a_c`` (in particular the unknown Bayes actions ``a*_c``), define

    S = (1/n) sum_c  min_a  sum_{i in c, retained} min(l(Y_i - a), kappa)        (observable)
    W_g = (G/n) sum_{i in g, retained} min(l(Y_i - a*_{c(i)}), kappa)            (unobservable)

Then (i) S <= (1/G) sum_g W_g pointwise, (ii) the W_g are independent with
W_g in [0, b_g], b_g = kappa * G * m_g / n (m_g = retained cases of cluster g), and
(iii) E[(1/G) sum_g W_g] <= R := (1/n) sum_i R_{c(i)} (case-weighted Bayes risk), because
truncation and dropping cases only remove nonnegative loss.  Hence for every t,

    P(S - R >= t) <= P( mean_g W_g - E mean_g W_g >= t ),

so ANY one-sided upper-tail inequality for independent bounded summands that needs only
the observed mean and the ranges -- Hoeffding (heterogeneous ranges) or Bentkus (common
range) -- inverts into a lower confidence endpoint for R.  Variance-adaptive inequalities
(empirical Bernstein, betting) are NOT licensed: the W_g are not observed.

The statistic is computed exactly: the truncated objective is piecewise linear in ``a``
with kinks at data points and their cap shifts, so its global minimum is attained at a
kink; every kink is evaluated with prefix sums (O(n log n) per cell).  An approximate
minimiser would be INVALID (it could exceed the value at a*), so none is used.

This is the classical SAA optimism inequality used pathwise rather than in expectation;
the contribution is its support-free, cluster-valid, full-sample use inside Pythia.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


# ------------------------------------------------------------------ exact truncated SAA
def _truncated_pinball_min(values: np.ndarray, q: float, kappa: float) -> float:
    """exact min_a sum_i min(rho_q(y_i - a), kappa); MAE is q = 1/2 with loss 2*rho."""
    y = np.sort(np.asarray(values, float))
    n = len(y)
    if n == 0:
        return 0.0
    P = np.concatenate([[0.0], np.cumsum(y)])
    up, dn = kappa / q, kappa / (1 - q)          # y - a >= up  or  a - y >= dn  => truncated
    cand = np.unique(np.concatenate([y, y - up, y + dn]))
    lo = np.searchsorted(y, cand - dn, side="right")      # y > a - dn
    mid = np.searchsorted(y, cand, side="left")           # y < a
    hi = np.searchsorted(y, cand + up, side="left")       # y < a + up
    below = (1 - q) * (cand * (mid - lo) - (P[mid] - P[lo]))
    above = q * ((P[hi] - P[mid]) - cand * (hi - mid))
    outside = n - (hi - lo)
    f = below + above + kappa * outside
    return float(np.min(f))


def _window_miss_min(values: np.ndarray, window: float) -> float:
    y = np.sort(np.asarray(values, float))
    if len(y) == 0:
        return 0.0
    right = np.searchsorted(y, y + 2 * window, side="right")
    best = int(np.max(right - np.arange(len(y))))
    return float(len(y) - best)


def truncated_cell_sum(values, *, loss: str, kappa: float, quantile: float | None = None,
                       window: float | None = None) -> float:
    """exact min_a sum_i min(loss_i(a), kappa) for one cell."""
    v = np.asarray(values, float)
    if loss == "mae":
        return 2.0 * _truncated_pinball_min(v, 0.5, kappa / 2.0)
    if loss == "pinball":
        return _truncated_pinball_min(v, float(quantile), kappa)
    if loss == "window_miss":
        return _window_miss_min(v, float(window))     # loss already in [0,1]
    if loss == "zero_one":                            # categorical labels coded as numbers
        _, cnt = np.unique(v, return_counts=True)
        return float(len(v) - cnt.max())
    raise ValueError("loss must be mae, pinball, window_miss or zero_one")


# ------------------------------------------------------------------ inversions
def bentkus_lower_from_statistic(stat: float, n_units: int, common_range: float,
                                 alpha: float) -> float:
    """LCB on mu := mean_g E W_g when stat <= mean_g W_g and W_g in [0, B] independent.
    Inverts Bentkus (2004): P(sum W/B >= x) <= e * P(Bin(n, mu/B) >= x) (log-linear interp.)."""
    if stat <= 0:
        return 0.0
    B = float(common_range)
    p_obs = min(1.0, stat / B)
    x = n_units * p_obs
    k = min(int(math.floor(x)), n_units)
    frac = x - k

    def log_tail(p):
        if p <= 0:
            return -np.inf
        if p >= 1:
            return 0.0
        low = float(binom.logsf(k - 1, n_units, p))
        if frac < 1e-12 or k >= n_units:
            return low
        high = float(binom.logsf(k, n_units, p))
        return (1 - frac) * low + frac * high

    target = math.log(alpha) - 1.0
    if log_tail(p_obs) < target:      # numerically degenerate; be conservative
        return 0.0
    if log_tail(1e-300) >= target:
        return 0.0
    root = brentq(lambda p: log_tail(p) - target, 1e-300, p_obs, xtol=1e-14)
    return float(B * root)


def hoeffding_lower_from_statistic(stat: float, ranges: np.ndarray, alpha: float) -> float:
    G = len(ranges)
    return float(max(0.0, stat - math.sqrt(math.log(1 / alpha) * float(np.sum(ranges ** 2)) / (2 * G * G))))


# ------------------------------------------------------------------ certificate
@dataclass(frozen=True)
class DominatedSAACertificate:
    status: str
    claim_tier: str
    loss: str
    bound: float | None
    selected_cap: float | None
    selected_inequality: str | None
    alpha_family: float
    alpha_per_candidate: float | None
    candidates: tuple = ()
    assumptions: tuple = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    refusal: Mapping[str, str] | None = None

    def to_dict(self):
        return asdict(self)


def dominated_saa_lower_bound(
        cell_labels, outcomes, clusters, *, loss: str, caps: Sequence[float],
        alpha: float = .05, quantile: float | None = None, window: float | None = None,
        inequalities: Sequence[str] = ("bentkus", "hoeffding"),
        max_cases_per_cluster: int | None = None,
        retention_grid: Sequence[int | None] | None = None,
        representation_provenance: str, independent_clusters_attested: bool,
        outcome_blind_labels_attested: bool, case_retention_outcome_blind: bool = True,
        exploratory_cluster_override: bool = False) -> DominatedSAACertificate:
    """Support-free, cluster-valid, full-sample lower endpoint for case-weighted Bayes risk.

    ``max_cases_per_cluster`` (applied in the given row order, which must be outcome-blind)
    limits each cluster's range; dropped cases contribute a safe zero.  Alpha is split by
    Bonferroni over every (cap, inequality) candidate; the reported bound is their maximum.
    """
    labels = np.asarray(cell_labels).astype(str).reshape(-1)
    y = np.asarray(outcomes, float).reshape(-1)
    group = np.asarray(clusters).astype(str).reshape(-1)
    sel = str(loss).lower()
    n = len(y)
    if n < 2 or len(labels) != n or len(group) != n or not np.isfinite(y).all() or not 0 < alpha < 1:
        raise ValueError("aligned finite rows, labels, clusters and valid alpha required")
    if sel not in {"mae", "pinball", "window_miss"}:
        raise ValueError("loss must be mae, pinball, or window_miss")
    if sel == "pinball" and (quantile is None or not 0 < quantile < 1):
        raise ValueError("pinball requires a quantile in (0,1)")
    if sel == "window_miss" and (window is None or window <= 0):
        raise ValueError("window_miss requires a positive half-window")
    if not representation_provenance:
        raise ValueError("representation provenance is required")
    ineqs = tuple(str(v).lower() for v in inequalities)
    if not ineqs or any(v not in {"bentkus", "hoeffding"} for v in ineqs):
        raise ValueError("inequalities must be bentkus and/or hoeffding")
    caps = tuple(float(c) for c in caps) if sel != "window_miss" else (1.0,)
    if not caps or any((not math.isfinite(c)) or c <= 0 for c in caps):
        raise ValueError("positive finite caps required")
    missing = []
    if not independent_clusters_attested and not exploratory_cluster_override:
        missing.append("independent clusters")
    if not outcome_blind_labels_attested:
        missing.append("outcome-blind representation labels")
    if not case_retention_outcome_blind:
        missing.append("outcome-blind case retention order")
    if missing:
        return DominatedSAACertificate("refused", "refused", sel, None, None, None, float(alpha), None,
                                       refusal={"reason": "missing required contract: " + ", ".join(missing)})

    uniq, ginv = np.unique(group, return_inverse=True)
    G = len(uniq)
    if retention_grid is None:
        retention_grid = (max_cases_per_cluster,)
    elif max_cases_per_cluster is not None:
        raise ValueError("pass either max_cases_per_cluster or retention_grid, not both")
    retention_grid = tuple(retention_grid)
    if not retention_grid or any(m is not None and int(m) < 1 for m in retention_grid):
        raise ValueError("retention values must be positive integers or None")
    # outcome-blind within-cluster rank in the given row order
    rank = np.zeros(n, int)
    seen = np.zeros(G, int)
    for i in range(n):
        seen[ginv[i]] += 1
        rank[i] = seen[ginv[i]]

    n_cand = len(caps) * len(ineqs) * len(retention_grid)
    a_each = alpha / n_cand
    cands = []
    for m in retention_grid:
        keep = np.ones(n, bool) if m is None else rank <= int(m)
        m_g = np.bincount(ginv[keep], minlength=G).astype(float)
        for kappa in caps:
            total = 0.0
            for c in np.unique(labels[keep]):
                total += truncated_cell_sum(y[keep & (labels == c)], loss=sel, kappa=kappa,
                                            quantile=quantile, window=window)
            stat = total / n
            ranges = kappa * G * m_g / n
            for ineq in ineqs:
                if ineq == "bentkus":
                    b = bentkus_lower_from_statistic(stat, G, float(ranges.max()), a_each)
                else:
                    b = hoeffding_lower_from_statistic(stat, ranges, a_each)
                cands.append({"cap": kappa, "inequality": ineq, "retention": m, "statistic": stat,
                              "bound": b, "retained_rows": int(keep.sum())})
    best = max(cands, key=lambda d: d["bound"])
    tier = "conditional" if independent_clusters_attested else "exploratory"
    return DominatedSAACertificate(
        status=tier, claim_tier=tier, loss=sel, bound=float(best["bound"]),
        selected_cap=best["cap"], selected_inequality=best["inequality"],
        alpha_family=float(alpha), alpha_per_candidate=a_each, candidates=tuple(cands),
        assumptions=(
            "independent entity clusters; arbitrary dependence within a cluster",
            "representation cells frozen without outcomes; retention order outcome-blind",
            "fixed-action domination: S <= mean of independent bounded unobserved W_g",
            "only range-based inequalities (Hoeffding, Bentkus) are used",
            "no population outcome support required: losses are truncated, never outcomes",
        ),
        diagnostics={"n_rows": n, "n_clusters": G, "retained_rows": best["retained_rows"],
                     "selected_retention": best["retention"],
                     "representation": representation_provenance,
                     "model_training_performed": False})
