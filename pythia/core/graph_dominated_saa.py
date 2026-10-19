"""Dependency-graph dominated SAA: support-free Bayes-floor certificates without iid,
without independent clusters, and without stationarity.

Setting.  Cases i = 1..n have frozen outcome-blind cells c(i) and declared dependency units
(e.g. patient, surgeon-day, calendar day).  Two cases may be dependent only if they share at
least one declared unit value (dependency-graph assumption, Janson 2004).  Laws may differ
arbitrarily across cases (time drift, case mix), i.e. NO stationarity is assumed.

Estimand (realized-period floor).  For each cell c, R_c = min_a (1/|c|) sum_{i in c} E l(Y_i - a):
the best error of ANY rule mapping the registered information to an action, applied over the
realized cases of the period.  Adding time windows to the cells makes the rule time-adaptive.
This is exactly the error floor of a deployed rule over the period, and it needs no
stationarity of P(X) or P(Y|X).

Theorem (graph-Chernoff domination).  Let a_c be the minimisers above (unknown, FIXED) and
W_i = min(l(Y_i - a_{c(i)}), kappa) in [0, kappa].  With S the exact truncated cellwise SAA mean,
  (i)  S <= Wbar := (1/n) sum_i W_i  pathwise;   (ii) E Wbar <= R := (1/n) sum_c |c| R_c.
Let a proper colouring of the dependency graph have K colour classes (independent sets).
Generalised Hoelder over the K classes, independence inside a class, the Bernoulli extremality
of [0,kappa]-variables (Hoeffding 1963) and concavity of log(1 + p x) give, for every lambda>0,
  E exp(lambda n Wbar) <= (1 + pbar (e^{lambda K kappa} - 1))^{n/K},   pbar = E Wbar / kappa,
i.e. n Wbar is dominated in MGF by kappa*K*Binomial(n/K, pbar).  Chernoff:
  P(S >= s) <= P(Wbar >= s) <= exp(-(n/K) KL(s/kappa || pbar)),   s/kappa >= pbar.
Inverting at the observed S gives a lower confidence endpoint for pbar*kappa <= R.
K = 1 recovers row-independent Chernoff; a patient-only graph gives K = max cases/patient.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Mapping, Sequence

import numpy as np

from .dominated_saa import truncated_cell_sum


# ---------------------------------------------------------------- colouring
def dependency_colouring(units: Mapping[str, Sequence], n: int):
    """Greedy (largest-unit-first) proper colouring of the graph whose cliques are the
    declared units.  Returns colours (n,), K, and the max clique size omega (a lower bound
    on any colouring; K == omega certifies optimality)."""
    groups = []
    for name, values in units.items():
        v = np.asarray(values).astype(str).reshape(-1)
        if len(v) != n:
            raise ValueError(f"unit {name} misaligned")
        _, inv = np.unique(v, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(inv[order])) + 1
        for g in np.split(order, bounds):
            if len(g) > 1:
                groups.append(g)
    omega = max([len(g) for g in groups], default=1)
    member = [[] for _ in range(n)]
    for gi, g in enumerate(groups):
        for i in g:
            member[i].append(gi)
    degree = np.zeros(n, int)
    for g in groups:
        degree[g] += len(g) - 1
    colour = np.full(n, -1, int)
    for i in np.argsort(-degree, kind="stable"):
        used = set()
        for gi in member[i]:
            for j in groups[gi]:
                if colour[j] >= 0:
                    used.add(colour[j])
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    return colour, int(colour.max() + 1), int(omega)


# ---------------------------------------------------------------- inversion
def _kl(q, p):
    q = min(max(q, 0.0), 1.0)
    p = min(max(p, 1e-300), 1 - 1e-16)
    a = 0.0 if q == 0 else q * math.log(q / p)
    b = 0.0 if q == 1 else (1 - q) * math.log((1 - q) / (1 - p))
    return a + b


def chernoff_kl_lower(stat_mean: float, cap: float, n_eff: float, alpha: float) -> float:
    """inf{mu : (n_eff) KL(stat/cap || mu/cap) < log(1/alpha)}  (one-sided upper-tail inversion)."""
    q = min(1.0, max(0.0, stat_mean / cap))
    if q <= 0:
        return 0.0
    thr = math.log(1 / alpha) / n_eff
    if _kl(q, 1e-300) < thr:
        return 0.0
    lo, hi = 0.0, q          # KL(q||p) decreasing in p on (0, q]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _kl(q, mid) >= thr:
            lo = mid
        else:
            hi = mid
    return float(cap * lo)


# ---------------------------------------------------------------- certificate
@dataclass(frozen=True)
class GraphDominatedCertificate:
    status: str
    claim_tier: str
    loss: str
    bound: float | None
    selected_cap: float | None
    selected_retention: int | None
    alpha_family: float
    alpha_per_candidate: float | None
    candidates: tuple = ()
    assumptions: tuple = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    refusal: Mapping[str, str] | None = None

    def to_dict(self):
        return asdict(self)


def graph_dominated_saa_lower_bound(
        cell_labels, outcomes, dependency_units: Mapping[str, Sequence], *, loss: str,
        caps: Sequence[float], alpha: float = .05, quantile: float | None = None,
        window: float | None = None, retention_grid: Sequence[int | None] = (None,),
        retention_unit: str | None = None, representation_provenance: str,
        dependency_graph_attested: bool, outcome_blind_labels_attested: bool,
        exploratory_override: bool = False) -> GraphDominatedCertificate:
    """Lower endpoint for the realized-period Bayes floor under a declared dependency graph.

    ``retention_grid`` keeps the first m cases (row order, outcome-blind) of every value of
    ``retention_unit`` (default: the first declared unit), shrinking cliques and hence K at the
    safe cost of dropped (zero-contribution) cases.  Every (retention, cap) candidate is charged.
    """
    lab = np.asarray(cell_labels).astype(str).reshape(-1)
    y = np.asarray(outcomes, float).reshape(-1)
    n = len(y)
    sel = str(loss).lower()
    if n < 2 or len(lab) != n or not np.isfinite(y).all() or not 0 < alpha < 1:
        raise ValueError("aligned finite rows and valid alpha required")
    if sel not in {"mae", "pinball", "window_miss"}:
        raise ValueError("loss must be mae, pinball or window_miss")
    if not dependency_units:
        raise ValueError("declare at least one dependency unit (use a row id for independent rows)")
    if not representation_provenance:
        raise ValueError("representation provenance required")
    missing = []
    if not dependency_graph_attested and not exploratory_override:
        missing.append("dependency graph (dependence only through declared units)")
    if not outcome_blind_labels_attested:
        missing.append("outcome-blind representation labels")
    if missing:
        return GraphDominatedCertificate("refused", "refused", sel, None, None, None, float(alpha), None,
                                         refusal={"reason": "missing required contract: " + ", ".join(missing)})
    caps = (1.0,) if sel == "window_miss" else tuple(float(c) for c in caps)
    units = {k: np.asarray(v).astype(str).reshape(-1) for k, v in dependency_units.items()}
    r_unit = retention_unit or next(iter(units))
    _, rinv = np.unique(units[r_unit], return_inverse=True)
    rank = np.zeros(n, int)
    seen = np.zeros(rinv.max() + 1, int)
    for i in range(n):
        seen[rinv[i]] += 1
        rank[i] = seen[rinv[i]]
    grid = tuple(retention_grid)
    a_each = alpha / (len(caps) * len(grid))
    cands = []
    for m in grid:
        keep = np.ones(n, bool) if m is None else rank <= int(m)
        idx = np.flatnonzero(keep)
        colour, K, omega = dependency_colouring({k: v[idx] for k, v in units.items()}, len(idx))
        for kappa in caps:
            total = 0.0
            for c in np.unique(lab[idx]):
                total += truncated_cell_sum(y[idx][lab[idx] == c], loss=sel, kappa=kappa,
                                            quantile=quantile, window=window)
            stat = total / n          # dropped cases contribute zero to the n-normalised mean
            # n Wbar sums n terms in [0,kappa], dropped ones identically 0: use n_eff = kept/K and
            # rescale the statistic to the kept-case mean.
            n_keep = len(idx)
            lower_keep = chernoff_kl_lower(stat * n / n_keep, kappa, n_keep / K, a_each)
            b = lower_keep * n_keep / n
            cands.append({"retention": m, "cap": kappa, "colours": K, "max_clique": omega,
                          "kept_rows": int(n_keep), "statistic": stat, "bound": b})
    best = max(cands, key=lambda d: d["bound"])
    tier = "conditional" if dependency_graph_attested else "exploratory"
    return GraphDominatedCertificate(
        status=tier, claim_tier=tier, loss=sel, bound=float(best["bound"]),
        selected_cap=best["cap"], selected_retention=best["retention"], alpha_family=float(alpha),
        alpha_per_candidate=a_each, candidates=tuple(cands),
        assumptions=(
            "dependence only between cases sharing a declared unit (dependency graph)",
            "cells and retention order frozen without outcomes",
            "estimand: realized-period floor of rules measurable in the registered cells; no stationarity",
            "no outcome support: losses truncated, never outcomes; no model training",
        ),
        diagnostics={"n_rows": n, "units": list(units), "representation": representation_provenance,
                     "selected_colours": best["colours"], "selected_max_clique": best["max_clique"],
                     "model_training_performed": False})


# ---------------------------------------------------------------- dependence audit (development data only)
def dependence_audit(dev_outcomes, dev_cell_labels, candidate_units: Mapping[str, Sequence],
                     n_perm: int = 999, seed: int = 0):
    """Outcome-USING audit, to be run on a development split only.  For each candidate unit,
    tests whether cell-median-centred outcomes are more similar within unit values than under
    within-cell permutation (statistic: between-unit variance of unit means).  Units that
    reject should be declared in the certification graph."""
    y = np.asarray(dev_outcomes, float)
    lab = np.asarray(dev_cell_labels).astype(str)
    resid = y.copy()
    for c in np.unique(lab):
        m = lab == c
        resid[m] = y[m] - np.median(y[m])
    rng = np.random.default_rng(seed)
    out = {}
    for name, values in candidate_units.items():
        v = np.asarray(values).astype(str)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        multi = cnt[inv] > 1
        if multi.sum() < 4:
            out[name] = {"p_value": None, "reason": "too few repeated units"}
            continue

        def stat(r):
            s = np.bincount(inv, weights=r, minlength=len(cnt))
            return float(np.sum(s[cnt > 1] ** 2 / cnt[cnt > 1]))
        obs = stat(resid)
        cells = [np.flatnonzero(lab == c) for c in np.unique(lab)]
        ge = 0
        for _ in range(n_perm):
            r = resid.copy()
            for ix in cells:
                r[ix] = r[rng.permutation(ix)]
            ge += stat(r) >= obs
        out[name] = {"p_value": (1 + ge) / (1 + n_perm), "statistic": obs,
                     "repeated_values": int((cnt > 1).sum())}
    return out


def dependence_sensitivity_curve(statistic_mean: float, cap: float, n_rows: int,
                                 inflations: Sequence[float], alpha: float) -> list[dict]:
    """Bound as a function of an assumed dependence inflation K (effective sample n/K).
    Because the Chernoff endpoint is valid for any K >= the true fractional chromatic number,
    reporting the whole curve (and the tipping K at which the bound hits a threshold) turns the
    undeclared-dependence assumption into an explicit sensitivity parameter."""
    return [{"K": float(K), "bound": chernoff_kl_lower(statistic_mean, cap, n_rows / K, alpha)}
            for K in inflations]
