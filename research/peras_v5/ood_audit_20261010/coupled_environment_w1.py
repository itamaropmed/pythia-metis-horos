"""Outcome-only W1 caps from independently repeated joint environments.

Each environment supplies one coupled outcome draw for every registered target
unit. The within-environment dependence is arbitrary; only unitwise marginals
must equal the target conditional laws and environment rows must be independent.
No predictive model or feature smoothness is used.
"""
from __future__ import annotations

import numpy as np

from peras.core.concentration import bentkus_lower_mean


METHODS = ("Hoeffding", "empirical_Bernstein", "Bentkus")


def _upper_mean(values: np.ndarray, value_range: float, alpha: float,
                method: str) -> float:
    values = np.asarray(values, float).ravel()
    n = len(values)
    if n < 2 or not 0 < alpha < 1 or value_range <= 0:
        raise ValueError("at least two bounded values, positive range, and alpha required")
    if np.any(values < -1e-12) or np.any(values > value_range + 1e-12):
        raise ValueError("values exceed the declared range")
    if method == "Bentkus":
        return float(value_range - bentkus_lower_mean(
            value_range - values, value_range, alpha))
    if method == "Hoeffding":
        radius = value_range * np.sqrt(np.log(1 / alpha) / (2 * n))
    elif method == "empirical_Bernstein":
        log_term = np.log(2 / alpha)
        radius = (np.sqrt(2 * np.var(values, ddof=1) * log_term / n)
                  + 7 * value_range * log_term / (3 * (n - 1)))
    else:
        raise ValueError("unsupported method")
    return float(min(value_range, np.mean(values) + radius))


def environment_costs(replicates: np.ndarray, blocks: np.ndarray, *,
                      clip: tuple[float, float], shift: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Average ordered within-block absolute cost in each environment."""
    y = np.asarray(replicates, float)
    idx = np.asarray(blocks, int)
    lo, hi = map(float, clip)
    if (y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 2
            or not np.isfinite(y).all() or not np.isfinite([lo, hi]).all()
            or lo >= hi):
        raise ValueError("finite replicate matrix and ordered finite clip required")
    if (idx.ndim != 2 or idx.shape[0] < 1 or idx.shape[1] < 2
            or np.any(idx < 0) or np.any(idx >= y.shape[1])
            or np.unique(idx).size != idx.size):
        raise ValueError("globally disjoint valid blocks required")
    if shift is not None:
        s = np.asarray(shift, float).ravel()
        if len(s) != y.shape[1] or not np.isfinite(s).all():
            raise ValueError("finite unit-aligned shift required")
        y = y - s[None, :]
    y = np.clip(y, lo, hi)
    k = idx.shape[1]
    costs = np.zeros(y.shape[0])
    for a in range(k):
        left = y[:, idx[:, a]]
        for b in range(k):
            if a != b:
                costs += np.mean(np.abs(left - y[:, idx[:, b]]), axis=1) / (k * k)
    value_range = (hi - lo) * (k - 1) / k
    if np.any(costs > value_range + 1e-9):
        raise AssertionError("environment cost exceeded its deterministic range")
    return costs, float(value_range)


def calibrate_coupled_mean_w1(
        replicates: np.ndarray, blocks: np.ndarray, *,
        clip: tuple[float, float], delta_family: float = .025,
        methods: tuple[str, ...] = METHODS,
        coupling_provenance: str | None = None,
        shift: np.ndarray | None = None,
        shift_provenance: str | None = None) -> dict:
    """Upper-confidence cap on average within-block marginal W1.

    For every edge (i,j), W1(Q_i,Q_j) is no larger than the expected absolute
    cost under any coupling with marginals Q_i,Q_j. Averaging preserves the
    inequality. Environment-level concentration then upper-bounds that mean.
    """
    if not 0 < delta_family < 1 or not methods or len(set(methods)) != len(methods):
        raise ValueError("valid family delta and unique methods required")
    if coupling_provenance is None or not coupling_provenance.strip():
        return {"status": "refused", "mean_w1_cap": None,
                "reason": "coupling provenance required"}
    if shift is not None and (shift_provenance is None or not shift_provenance.strip()):
        return {"status": "refused", "mean_w1_cap": None,
                "reason": "shift provenance required"}
    costs, value_range = environment_costs(
        replicates, blocks, clip=clip, shift=shift)
    alpha = delta_family / len(methods)
    candidates = [{"method": method,
                   "upper": _upper_mean(costs, value_range, alpha, method)}
                  for method in methods]
    chosen = min(candidates, key=lambda row: row["upper"])
    return {
        "status": "conditional", "claim_tier": "conditional",
        "mean_w1_cap": float(chosen["upper"]),
        "selected_method": chosen["method"],
        "candidates": candidates,
        "empirical_coupling_cost": float(np.mean(costs)),
        "environment_cost_sd": float(np.std(costs, ddof=1)),
        "environment_cost_range": float(value_range),
        "n_environments": int(len(costs)),
        "delta_family": float(delta_family), "alpha_per_method": float(alpha),
        "clip": list(map(float, clip)),
        "coupling_provenance": coupling_provenance,
        "shift_provenance": "none" if shift is None else shift_provenance,
        "target": "average ordered within-block W1 of clipped aligned unit-conditional laws",
        "assumptions": [
            "each environment row has the registered unit-conditional laws as its coordinate marginals",
            "environment rows are independent and identically distributed",
            "blocks, clip, shift, methods, and alpha allocation were fixed without replicate outcomes",
            "the evaluation outcomes used by the downstream risk witness are independent of these environments",
        ],
        "warning": "ordinary unmatched observational rows are not repeated joint environments",
    }
