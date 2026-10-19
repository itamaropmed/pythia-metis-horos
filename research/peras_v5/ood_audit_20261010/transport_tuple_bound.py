"""Conditional fixed-design lower bound for outcome-blind near-replicate tuples.

This module implements the already-sketched Wasserstein tuple-stability idea
with an explicit overlapping-tuple concentration margin.  The caller must
provide independently justified pairwise W1 upper bounds.  Synthetic oracle
caps are useful for falsification but do not certify a real-data analysis.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core import erm_risk


def lower_bound(
    outcomes: np.ndarray,
    tuples: np.ndarray,
    transport_caps: np.ndarray,
    *,
    loss: str,
    clip: tuple[float, float],
    alpha: float = 0.05,
    cap_provenance: str,
) -> dict[str, Any]:
    """Bound average conditional Bayes risk at the tuple anchors.

    Conditional on the frozen covariates, tuples, and caps, independent entity
    outcomes are assumed.  For each row i, cap[i,j] >= W1(P_anchor,P_member).
    Clipping contracts W1 and cannot increase the Bayes risk for the supported
    location losses.  The returned bound therefore concerns original outcomes.
    """
    y = np.asarray(outcomes, float).ravel()
    idx = np.asarray(tuples, int)
    cap = np.asarray(transport_caps, float)
    if y.size < 2 or idx.ndim != 2 or idx.shape[0] < 2 or idx.shape[1] < 2:
        raise ValueError("at least two entities and two tuples of size >=2 are required")
    if cap.shape != idx.shape or not np.isfinite(cap).all() or np.any(cap < 0):
        raise ValueError("transport caps must be finite, nonnegative, and tuple-aligned")
    if np.any(idx < 0) or np.any(idx >= y.size) or not np.isfinite(y).all():
        raise ValueError("outcomes and tuple indices must be finite and aligned")
    if np.any([np.unique(row).size != row.size for row in idx]):
        raise ValueError("an entity may not repeat within a tuple")
    lo, hi = map(float, clip)
    if not np.isfinite([lo, hi]).all() or lo >= hi or not 0 < alpha < 1:
        raise ValueError("finite ordered clip endpoints and alpha in (0,1) are required")
    if not cap_provenance.strip():
        raise ValueError("the source of transport caps must be declared")
    diameter = hi - lo
    if loss == "MAE":
        outcome_lipschitz = 1.0
    elif loss == "Q90":
        outcome_lipschitz = 0.9
    elif loss == "MSE":
        outcome_lipschitz = 2.0 * diameter
    else:
        return {"status": "refused", "bound": None,
                "reason": "W1 stability does not cover this loss; window losses require TV or a smoothed surrogate"}

    m, k = idx.shape
    yc = np.clip(y, lo, hi)
    statistic = float(np.mean(erm_risk(loss, yc[idx])))
    transport_penalty = outcome_lipschitz * float(np.mean(cap))
    incidence = np.bincount(idx.ravel(), minlength=y.size)
    effective_entities = float((m * k) ** 2 / np.sum(incidence.astype(float) ** 2))
    radius = (
        outcome_lipschitz * diameter / (m * k)
        * math.sqrt(0.5 * math.log(1.0 / alpha) * float(np.sum(incidence.astype(float) ** 2)))
    )
    raw = statistic - transport_penalty - radius
    return {
        "status": "conditional", "loss": loss, "bound": max(0.0, raw),
        "raw_bound": raw, "statistic": statistic,
        "transport_penalty": transport_penalty, "concentration_radius": radius,
        "effective_entities": effective_entities, "n_entities": y.size,
        "n_tuples": m, "tuple_size": k, "clip": [lo, hi], "alpha": alpha,
        "cap_provenance": cap_provenance,
        "target": "mean conditional Bayes risk at the tuple anchors, given the audited covariates",
        "assumptions": [
            "tuple indices and transport caps were frozen without evaluation outcomes",
            "entity outcomes are independent conditional on audited covariates",
            "each cap upper-bounds the original conditional-law W1 distance",
        ],
    }


def average_w1_sensitivity(
    outcomes: np.ndarray,
    tuples: np.ndarray,
    *,
    loss: str,
    clip: tuple[float, float],
    mean_w1_grid: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Diagnostic curve for an externally supplied *average* W1 cap.

    The transport proof is linear in the mean of the pairwise W1 distances.
    It therefore needs only a cap on that average, not valid individual caps.
    This function does not establish such a cap from outcomes; it returns a
    sensitivity curve, not an unconditional certificate. The same one-sided
    concentration event covers every value on the curve simultaneously.
    """
    grid = np.asarray(mean_w1_grid, float).ravel()
    if (grid.size == 0 or not np.isfinite(grid).all() or np.any(grid < 0)
            or np.any(np.diff(grid) < 0)):
        raise ValueError("mean_w1_grid must be nonempty, finite, nonnegative, and sorted")
    idx = np.asarray(tuples, int)
    # Zero transport caps are used only to reuse the validated statistic and
    # overlap-aware radius; the temporary conditional status is never exposed.
    base = lower_bound(
        outcomes, idx, np.zeros(idx.shape, dtype=float),
        loss=loss, clip=clip, alpha=alpha,
        cap_provenance="algebraic zero used only for sensitivity intercept",
    )
    if base["status"] == "refused":
        return base
    diameter = float(clip[1] - clip[0])
    lipschitz = {"MAE": 1.0, "Q90": 0.9, "MSE": 2.0 * diameter}[loss]
    intercept = base["statistic"] - base["concentration_radius"]
    bound = np.maximum(0.0, intercept - lipschitz * grid)
    breakpoint = max(0.0, intercept / lipschitz)
    return {
        "status": "sensitivity_only", "loss": loss,
        "mean_w1_grid": grid.tolist(), "bounds": bound.tolist(),
        "breakpoint": float(breakpoint), "statistic": base["statistic"],
        "concentration_radius": base["concentration_radius"],
        "outcome_lipschitz": lipschitz, "clip": base["clip"],
        "alpha": alpha, "n_tuples": base["n_tuples"],
        "tuple_size": base["tuple_size"],
        "target": base["target"],
        "assumption_to_interpret_point": (
            "the mean original-law W1 distance over all used tuple edges "
            "is no greater than the displayed grid value"
        ),
        "warning": "the grid is a sensitivity analysis, not a data-validated W1 cap",
    }
