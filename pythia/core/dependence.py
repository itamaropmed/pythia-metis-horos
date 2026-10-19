"""Dependence sensitivity for squared-loss disagreement certificates.

This module is deliberately diagnostic. It exposes the assumptions and feasible
moment geometry; it is not used automatically to upgrade a claim tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class DependencePoint:
    lambda_max: float
    bound: float
    psd_bound: float | None
    anchor_bound: float
    feasible: bool
    min_eigenvalue: float | None
    max_correlation: float | None
    feasible_risk_interval: tuple[float, float] | None
    active_constraints: tuple[str, ...]


def _moments(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    risks = np.mean((p - y[None, :]) ** 2, axis=1)
    d = np.zeros((p.shape[0], p.shape[0]))
    for i, j in combinations(range(p.shape[0]), 2):
        d[i, j] = d[j, i] = float(np.mean((p[i] - p[j]) ** 2))
    deltas = np.array([0.5 * d[i, j] for i, j in combinations(range(p.shape[0]), 2)])
    return risks, d, float(np.max(deltas))


def _covariance(risks: np.ndarray, disagreement: np.ndarray, risk_floor: float) -> np.ndarray:
    c = 0.5 * (risks[:, None] + risks[None, :] - disagreement) - risk_floor
    return 0.5 * (c + c.T)


def _feasible(risks: np.ndarray, disagreement: np.ndarray, risk_floor: float,
              lambda_max: float, tol: float) -> tuple[bool, float, float]:
    c = _covariance(risks, disagreement, risk_floor)
    eig = float(np.linalg.eigvalsh(c).min())
    diag = np.diag(c)
    if np.any(diag < -tol):
        return False, eig, float("inf")
    corr = []
    for i, j in combinations(range(c.shape[0]), 2):
        den = np.sqrt(max(diag[i], 0.0) * max(diag[j], 0.0))
        corr.append(c[i, j] / den if den > tol else (0.0 if c[i, j] <= tol else np.inf))
    max_corr = float(max(corr, default=0.0))
    return eig >= -tol and max_corr <= lambda_max + tol, eig, max_corr


def _minimum_feasible(risks: np.ndarray, disagreement: np.ndarray,
                      lambda_max: float, tol: float) -> tuple[float | None, dict]:
    upper = float(np.min(risks))
    grid = np.linspace(0.0, upper, 4097)
    states = [_feasible(risks, disagreement, value, lambda_max, tol) for value in grid]
    hits = [i for i, state in enumerate(states) if state[0]]
    if not hits:
        return None, {"reason": "no PSD moment matrix satisfies the correlation cap"}
    hit = hits[0]
    lo = grid[max(hit - 1, 0)]
    hi = grid[hit]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _feasible(risks, disagreement, mid, lambda_max, tol)[0]:
            hi = mid
        else:
            lo = mid
    ok, eig, corr = _feasible(risks, disagreement, hi, lambda_max, tol)
    upper_feasible = float(grid[hits[-1]])
    active = []
    if hi <= tol:
        active.append("nonnegative_risk_floor")
    if abs(eig) <= 10 * tol:
        active.append("positive_semidefinite_covariance")
    if abs(corr - lambda_max) <= 10 * tol:
        active.append("shared_error_correlation_cap")
    return float(hi), {
        "min_eigenvalue": eig, "max_correlation": corr, "feasible": ok,
        "feasible_risk_interval": (float(hi), upper_feasible),
        "active_constraints": tuple(active),
    }


def _minimum_required_lambda(risks: np.ndarray, disagreement: np.ndarray,
                             tol: float) -> float:
    """Smallest nonnegative correlation cap admitting any PSD risk floor."""
    candidates = []
    for risk_floor in np.linspace(0.0, float(np.min(risks)), 4097):
        c = _covariance(risks, disagreement, risk_floor)
        if float(np.linalg.eigvalsh(c).min()) < -tol or np.any(np.diag(c) < -tol):
            continue
        _, _, corr = _feasible(risks, disagreement, risk_floor, 1.0, tol)
        if np.isfinite(corr):
            candidates.append(max(0.0, corr))
    return float(min(candidates)) if candidates else float("nan")


def dependence_profile(y: np.ndarray, p: np.ndarray,
                       lambdas: np.ndarray | None = None,
                       tolerance: float = 1e-8) -> dict:
    """Return an exploratory B(lambda) profile for squared loss.

    The PSD component is the smallest risk compatible with the observed second
    moments and a cap on positive error correlation. The max-pair anchor makes
    lambda=0 exactly reproduce the currently declared orthogonality rung. Until
    its proof audit is complete, the combined result is an exploratory
    sensitivity analysis rather than a certified lower bound.
    """
    y = np.asarray(y, float).ravel()
    p = np.atleast_2d(np.asarray(p, float))
    if p.shape[1] != y.size or p.shape[0] < 2:
        raise ValueError("dependence profile needs at least two aligned predictors")
    lambdas = np.linspace(0.0, 0.95, 20) if lambdas is None else np.asarray(lambdas, float)
    if np.any((lambdas < 0) | (lambdas >= 1)) or np.any(np.diff(lambdas) < 0):
        raise ValueError("lambdas must be sorted in [0, 1)")
    risks, disagreement, delta_max = _moments(y, p)
    L = float(np.min(risks))
    orthogonal = max(L - delta_max, 0.0)
    points = []
    for lam in lambdas:
        psd, diag = _minimum_feasible(risks, disagreement, float(lam), tolerance)
        anchor = max(L - delta_max / (1.0 - float(lam)), 0.0)
        # Preserve the known lambda=0 rung and never tighten as dependence grows.
        if np.isclose(lam, 0.0):
            combined = orthogonal
        else:
            combined = anchor if psd is None else min(anchor, psd)
        points.append(DependencePoint(
            lambda_max=float(lam), bound=float(combined), psd_bound=psd,
            anchor_bound=float(anchor), feasible=psd is not None,
            min_eigenvalue=diag.get("min_eigenvalue"),
            max_correlation=diag.get("max_correlation"),
            feasible_risk_interval=diag.get("feasible_risk_interval"),
            active_constraints=diag.get("active_constraints", ()),
        ))
    # Numerical safety: enforce the defining monotonicity of a sensitivity curve.
    running = float("inf")
    serial = []
    for point in points:
        running = min(running, point.bound)
        row = point.__dict__.copy()
        row["bound"] = running
        serial.append(row)
    _, _, corr_at_zero = _feasible(risks, disagreement, 0.0, 1.0, tolerance)
    anchor_break = max(0.0, min(1.0, 1.0 - delta_max / L)) if L > 0 else 0.0
    lambda_break = float(min(anchor_break, max(0.0, corr_at_zero)))
    minimum_lambda = _minimum_required_lambda(risks, disagreement, tolerance)
    requested_feasible = any(row["feasible"] for row in serial)
    return {
        "status": "exploratory" if requested_feasible else "refused",
        "theorem_candidate": True,
        "used_for_certificate": False,
        "L": L,
        "orthogonality_bound": orthogonal,
        "delta_max": delta_max,
        "lambda_break": lambda_break,
        "feasible_dependence_interval": (
            [minimum_lambda, 1.0] if np.isfinite(minimum_lambda) else None
        ),
        "requested_grid_has_feasible_point": requested_feasible,
        "curve": serial,
        "assumptions": [
            "squared loss",
            "population moments approximated by OOF empirical moments",
            "positive-semidefinite model-error covariance",
            "pairwise positive error correlation bounded by lambda_max",
        ],
    }
