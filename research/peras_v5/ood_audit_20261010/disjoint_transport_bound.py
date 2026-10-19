"""Conditional Bayes-risk lower bound from disjoint near-replicate tuples.

Disjointness permits Bentkus concentration on bounded *tuple scores* rather
than worst-case bounded differences of all raw outcomes. A supplied upper
bound on mean within-tuple conditional-law W1 remains indispensable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core import erm_risk
from peras.core.concentration import bentkus_lower_mean


def lower_bound(
    outcomes: np.ndarray,
    blocks: np.ndarray,
    *,
    loss: str,
    mean_w1_cap: float,
    score_clip: float,
    alpha: float = 0.05,
    cap_provenance: str,
    outcome_clip: tuple[float, float] | None = None,
    shift: np.ndarray | None = None,
    shift_provenance: str | None = None,
) -> dict[str, Any]:
    """Bound full fixed-design risk using independent matched tuple scores.

    The external cap must upper-bound the average *ordered* within-block W1
    between the laws of C(Y_i-shift_i), or C(Y_i) with no shift, where C is
    the optional predeclared outcome clip (identity if absent):
    (m k²)^-1 sum_t sum_i sum_j W1(Q_ti,Q_tj). The diagonal terms are zero.
    Grouping and the cap must be fixed without the evaluation outcomes.
    Unused entities contribute a conservative zero to the full-design target.
    """
    y = np.asarray(outcomes, float).ravel()
    idx = np.asarray(blocks, int)
    if (y.size < 2 or idx.ndim != 2 or idx.shape[0] < 2
            or idx.shape[1] < 2 or not np.isfinite(y).all()):
        raise ValueError("finite outcomes and at least two blocks of size >=2 required")
    if np.any(idx < 0) or np.any(idx >= y.size) or np.unique(idx).size != idx.size:
        raise ValueError("blocks must have valid, globally disjoint entity indices")
    if (not np.isfinite(mean_w1_cap) or mean_w1_cap < 0
            or not np.isfinite(score_clip) or score_clip <= 0
            or not 0 < alpha < 1 or not cap_provenance.strip()):
        raise ValueError("finite nonnegative cap, positive clip, alpha, and provenance required")
    if outcome_clip is not None:
        lo, hi = map(float, outcome_clip)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("outcome clip must have finite ordered endpoints")
    if loss == "MAE":
        lipschitz = 1.0
    elif loss == "Q90":
        lipschitz = 0.9
    elif loss == "MSE":
        if outcome_clip is None:
            return {"status": "refused", "bound": None,
                    "reason": "MSE requires a finite outcome clip for W1 stability"}
        lipschitz = 2.0 * (hi - lo)
    else:
        return {"status": "refused", "bound": None,
                "reason": "loss is not W1-Lipschitz under this score construction"}
    if shift is None:
        aligned_y = y
        declared_shift = "none"
    else:
        s = np.asarray(shift, float).ravel()
        if (s.size != y.size or not np.isfinite(s).all()
                or shift_provenance is None or not shift_provenance.strip()):
            raise ValueError("aligned shift must be finite, entity-aligned, and provenance-declared")
        aligned_y = y - s
        declared_shift = shift_provenance

    yy = np.clip(aligned_y, lo, hi) if outcome_clip is not None else aligned_y
    raw_scores = erm_risk(loss, yy[idx])
    scores = np.minimum(raw_scores, score_clip)
    if np.any(scores < -1e-12) or not np.isfinite(scores).all():
        raise AssertionError("tuple ERM risk should be finite and nonnegative")
    lower_score_mean = bentkus_lower_mean(scores, score_clip, alpha)
    transport_penalty = lipschitz * mean_w1_cap
    retained_fraction = idx.size / y.size
    bound = retained_fraction * max(0.0, lower_score_mean - transport_penalty)
    return {
        "status": "conditional", "claim_tier": "conditional", "loss": loss,
        "bound": float(bound), "lower_score_mean": float(lower_score_mean),
        "score_mean": float(np.mean(scores)), "score_clip": float(score_clip),
        "transport_penalty": float(transport_penalty),
        "mean_w1_cap": float(mean_w1_cap),
        "tolerated_mean_w1": float(lower_score_mean / lipschitz),
        "outcome_lipschitz": float(lipschitz), "alpha": float(alpha),
        "n_entities": int(y.size), "n_blocks": int(idx.shape[0]),
        "tuple_size": int(idx.shape[1]), "retained_fraction": float(retained_fraction),
        "cap_provenance": cap_provenance,
        "shift_provenance": declared_shift,
        "outcome_clip": list(outcome_clip) if outcome_clip is not None else None,
        "target": "full audited fixed-design mean Bayes risk, with zero for omitted entities",
        "assumptions": [
            "blocks and cap were frozen without evaluation outcomes",
            "any supplied shift was frozen without evaluation outcomes",
            "entity outcomes are independent conditional on audited covariates",
            "mean_w1_cap upper-bounds ordered within-block mean aligned-law W1 after outcome clipping, if any",
            "clipping is monotone and contractive for the supported location losses, so clipped Bayes risk is no greater than original Bayes risk",
        ],
    }
