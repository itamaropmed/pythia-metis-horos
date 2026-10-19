"""Resource-matched, training-free grouped SAA reference construction.

Calibration/audit stationarity of the group-reference laws is required.
The target uses the GROUP representation only; no sufficiency claim for
richer raw features is automatic. Audit weights, clipping, and tuple size
must be fixed without calibration outcomes. No predictor is fitted/stored.
"""
from __future__ import annotations

import numpy as np

from core import erm_risk


def empirical_floor(y: np.ndarray, loss: str) -> float:
    if loss == "MSE":
        return float(np.var(y))
    if loss == "MAE":
        return float(np.mean(np.abs(y - np.median(y))))
    if loss == "Q90":
        a = np.quantile(y, 0.9, method="inverted_cdf")
        u = y - a
        return float(np.mean(np.maximum(0.9 * u, -0.1 * u)))
    raise ValueError("supported losses are MAE, Q90, MSE")


def grouped_floor(samples: dict, weights: dict, *, loss: str,
                  clip: tuple[float, float], score_clip: float,
                  k: int = 8, alpha: float = 0.05 / 3) -> dict:
    lo, hi = map(float, clip)
    if (not np.isfinite([lo, hi, score_clip]).all() or lo >= hi
            or score_clip <= 0 or not 0 < alpha < 1 or k < 2):
        raise ValueError("valid clip, score clip, alpha, and k required")
    if (not weights or any(not np.isfinite(w) or w < 0 for w in weights.values())
            or not np.isclose(sum(weights.values()), 1.0, atol=1e-10)):
        raise ValueError("nonnegative weights must sum to one")
    width = hi - lo
    score_range = {"MAE": width, "Q90": 0.9 * width, "MSE": width ** 2}
    if loss not in score_range:
        raise ValueError("supported losses are MAE, Q90, MSE")
    tuple_center = full_center = plugin = 0.0
    tuple_sum_squares = full_sum_squares = 0.0
    n_blocks = 0
    for group, weight in weights.items():
        if weight == 0:
            continue
        if group not in samples:
            raise ValueError(f"missing calibration group {group!r}")
        raw = np.asarray(samples[group], float).ravel()
        if len(raw) < k or not np.isfinite(raw).all():
            raise ValueError(f"group {group!r} needs >=k finite outcomes")
        yy = np.clip(raw, lo, hi)
        m = len(yy) // k
        score = np.minimum(erm_risk(loss, yy[:m * k].reshape(m, k)), score_clip)
        tuple_center += weight * float(score.mean())
        full_center += weight * empirical_floor(yy, loss)
        plugin += weight * empirical_floor(raw, loss)
        tuple_sum_squares += weight ** 2 / m
        full_sum_squares += weight ** 2 / len(yy)
        n_blocks += m
    tuple_radius = score_clip * np.sqrt(np.log(1 / alpha) / 2 * tuple_sum_squares)
    full_radius = score_range[loss] * np.sqrt(np.log(1 / alpha) / 2 * full_sum_squares)
    return {"loss": loss, "tuple_bound": float(max(0., tuple_center - tuple_radius)),
            "full_saa_bound": float(max(0., full_center - full_radius)),
            "plugin_original": float(plugin), "plugin_clipped": float(full_center),
            "tuple_center": float(tuple_center), "tuple_radius": float(tuple_radius),
            "full_radius": float(full_radius), "n_blocks": n_blocks,
            "k": int(k), "alpha": float(alpha), "claim_tier": "conditional",
            "outcome_clip": [lo, hi], "score_clip": float(score_clip),
            "target": "audit-weighted Bayes risk for the declared group representation only",
            "raw_feature_certificate": False,
            "assumptions": ["iid calibration draws from each declared group-reference law",
                            "calibration-to-audit group-reference law stationarity",
                            "iid independent calibration outcomes",
                            "weights and all method choices fixed without calibration outcomes",
                            "raw-feature sufficiency is not asserted by this result"]}
