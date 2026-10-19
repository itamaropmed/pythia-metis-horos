"""Sensitivity certificates under a declared barycentric residual-law modulus.

This module is deliberately separate from predictor fitting.  It consumes a
frozen shift and outcome-blind, disjoint blocks.  With one outcome per unit the
within-block conditional-law discrepancy is not identified, so the result is
a simultaneous curve in a *declared* upper bound on that discrepancy.

For MAE and every pinball loss, let

    P_k = average_{blocks,units} int |F_iu - Fbar_u| du.

For MSE replace the absolute deviation by its square and call it P2_k.  If the
declared value dominates the relevant P_k, then the barycentric coupling
argument gives

    full-cohort risk >= retained_k * (E score_k - c_loss P_k),

where c_MAE=1, c_pinball=1/2, and c_MSE=1.  A simultaneous lower confidence
bound over all registered block sizes, score caps, and concentration methods
therefore yields a selection-safe lower envelope.  dKelly is the published
fixed-time betting construction implemented in ``repeated_environment_saa``.
"""
from __future__ import annotations

import numpy as np

from core import erm_risk
from repeated_environment_saa import PUBLISHED_METHODS, _lower_mean


def loss_coefficient(loss: str) -> float:
    if loss == "MAE" or loss == "MSE":
        return 1.0
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if 0 < q < 1:
            return 0.5
    raise ValueError("supported losses are MAE, MSE, and Qxx pinball")


def loss_range(loss: str, clip: tuple[float, float]) -> float:
    lo, hi = map(float, clip)
    width = hi - lo
    if not np.isfinite([lo, hi]).all() or width <= 0:
        raise ValueError("finite increasing clip required")
    if loss == "MAE":
        return width
    if loss == "MSE":
        return width * width
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if 0 < q < 1:
            return max(q, 1 - q) * width
    raise ValueError("supported losses are MAE, MSE, and Qxx pinball")


def block_scores(outcomes, blocks, *, loss: str,
                 clip: tuple[float, float], shift=None) -> np.ndarray:
    """One clipped empirical-optimum score per disjoint unit block."""
    y = np.asarray(outcomes, float).ravel()
    idx = np.asarray(blocks, int)
    if (idx.ndim != 2 or idx.shape[1] < 2 or idx.shape[0] < 2
            or np.unique(idx).size != idx.size or np.min(idx) < 0
            or np.max(idx) >= len(y) or not np.isfinite(y).all()):
        raise ValueError("finite outcomes and at least two globally disjoint blocks required")
    if shift is not None:
        shift = np.asarray(shift, float).ravel()
        if len(shift) != len(y) or not np.isfinite(shift).all():
            raise ValueError("finite shift aligned with outcomes required")
        y = y - shift
    y = np.clip(y, *map(float, clip))
    block = y[idx]
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        action = np.quantile(block, q, axis=1, method="inverted_cdf")
        residual = block - action[:, None]
        return np.mean(np.maximum(q * residual, (q - 1) * residual), axis=1)
    return np.asarray(erm_risk(loss, block), float)


def declared_modulus_portfolio(
        outcomes, blocks_by_k: dict[int, np.ndarray], *, loss: str,
        score_caps: tuple[float, ...], clip: tuple[float, float],
        declared_discrepancy: float | dict[int, float], alpha_family: float = .05,
        shift=None, shift_provenance=None, block_provenance=None,
        methods: tuple[str, ...] = ("dKelly_betting",)) -> dict:
    """Return a selection-safe bound at one point on the sensitivity curve.

    ``declared_discrepancy`` is P for MAE/pinball and P2 for MSE.  A scalar
    must upper-bound the discrepancy for every candidate block design.  A
    mapping may instead register one upper bound per k; selection remains safe
    because every score endpoint shares the same preallocated alpha ledger.
    """
    if not block_provenance or (shift is not None and not shift_provenance):
        return {"status": "refused", "bound": None,
                "reason": "block provenance and, when used, shift provenance required"}
    if isinstance(declared_discrepancy, dict):
        if set(declared_discrepancy) != set(blocks_by_k):
            raise ValueError("a block-specific declaration is required for every k")
        declarations = {int(k): float(v) for k, v in declared_discrepancy.items()}
    else:
        declarations = {int(k): float(declared_discrepancy) for k in blocks_by_k}
    if (not blocks_by_k or not score_caps or not methods
            or len(set(blocks_by_k)) != len(blocks_by_k)
            or len(set(score_caps)) != len(score_caps)
            or len(set(methods)) != len(methods)
            or any(m not in PUBLISHED_METHODS for m in methods)
            or not 0 < alpha_family < 1
            or any(not np.isfinite(v) or v < 0 for v in declarations.values())):
        raise ValueError("valid unique portfolio and nonnegative declaration required")
    coefficient = loss_coefficient(loss)
    maximum = loss_range(loss, clip)
    alpha = alpha_family / (len(blocks_by_k) * len(score_caps) * len(methods))
    n = len(np.asarray(outcomes).ravel())
    candidates = []
    for k, blocks in blocks_by_k.items():
        raw = block_scores(outcomes, blocks, loss=loss, clip=clip, shift=shift)
        retained = np.asarray(blocks).size / n
        declaration = declarations[int(k)]
        for requested_cap in score_caps:
            cap = min(float(requested_cap), maximum)
            if not np.isfinite(cap) or cap <= 0:
                raise ValueError("positive finite score caps required")
            scores = np.minimum(raw, cap)
            for method in methods:
                lcb = _lower_mean(scores, cap, alpha, method)
                candidates.append({
                    "k": int(k), "method": method, "score_cap": float(cap),
                    "n_blocks": int(len(scores)), "retained_fraction": float(retained),
                    "score_mean": float(np.mean(scores)), "lower_score_mean": float(lcb),
                    "declared_discrepancy": declaration,
                    "bound": float(retained * max(0., lcb - coefficient * declaration)),
                })
    chosen = max(candidates, key=lambda row: row["bound"])
    return {
        "status": "conditional", "claim_tier": "conditional",
        "loss": loss, "bound": chosen["bound"],
        "declared_discrepancy": chosen["declared_discrepancy"],
        "declared_discrepancy_by_k": declarations,
        "discrepancy_units": "minutes_squared" if loss == "MSE" else "minutes",
        "penalty_coefficient": float(coefficient),
        "selected_k": chosen["k"], "selected_method": chosen["method"],
        "selected_score_cap": chosen["score_cap"],
        "selected_lower_score_mean": chosen["lower_score_mean"],
        "retained_fraction": chosen["retained_fraction"],
        "tipping_discrepancy": float(chosen["lower_score_mean"] / coefficient),
        "alpha_family": float(alpha_family), "alpha_per_candidate": float(alpha),
        "methods": list(methods), "candidate_ks": list(map(int, blocks_by_k)),
        "score_caps": list(map(float, score_caps)), "clip": list(map(float, clip)),
        "block_provenance": str(block_provenance),
        "shift_provenance": str(shift_provenance or "none"),
        "target": "full-cohort original-outcome Bayes risk under the registered loss",
        "assumptions": [
            "units are independent and blocks are globally disjoint and frozen without evaluation outcomes",
            "the frozen shift is external to Peras and independent of evaluation outcomes",
            "the declaration upper-bounds the relevant average barycentric residual-law discrepancy for every selected block design",
            "outcome clipping can only decrease the supported location-loss Bayes risk",
            "the complete candidate portfolio and alpha allocation were frozen",
        ],
        "candidates": candidates,
    }
