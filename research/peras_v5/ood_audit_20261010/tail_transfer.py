"""Transfer clipped group-risk intervals to original-outcome upper bounds.

Lower endpoints transfer automatically for supported location losses because
monotone clipping contracts loss. Upper transfer needs a tail correction.
Window loss needs only tail probability; unbounded magnitude losses require
an external support/penalty cap or variance cap and otherwise refuse.
"""
from __future__ import annotations

import numpy as np


def penalty_values(y: np.ndarray, *, loss: str,
                   clip: tuple[float, float]) -> np.ndarray:
    raw = np.asarray(y, float).ravel()
    lo, hi = map(float, clip)
    if not np.isfinite(raw).all() or not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("finite outcomes and ordered clip required")
    clipped = np.clip(raw, lo, hi)
    tail = np.abs(raw - clipped)
    if loss == "MAE":
        return tail
    if loss == "MSE":
        return tail ** 2 + 2 * (hi - lo) * tail
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        return max(q, 1 - q) * tail
    if loss.startswith("W") and float(loss[1:]) > 0:
        return (tail > 0).astype(float)
    raise ValueError("unsupported loss")


def tail_upper(samples: dict, weights: dict, *, loss: str,
               clip: tuple[float, float], alpha: float,
               penalty_cap: float | None = None,
               variance_caps: dict | None = None) -> dict:
    if (not 0 < alpha < 1 or not weights
            or any(not np.isfinite(w) or w < 0 for w in weights.values())
            or not np.isclose(sum(weights.values()), 1., atol=1e-10)):
        raise ValueError("valid alpha and nonnegative weights summing to one required")
    arrays = {}
    mean = 0.
    for group, weight in weights.items():
        if weight == 0:
            continue
        if group not in samples:
            return {"status": "refused", "reason": "missing positive-weight tail group"}
        values = penalty_values(samples[group], loss=loss, clip=clip)
        if len(values) < 2:
            return {"status": "refused", "reason": "tail group needs at least two outcomes"}
        arrays[group] = values
        mean += weight * float(values.mean())
    if loss.startswith("W"):
        penalty_cap = 1.
    if penalty_cap is not None and variance_caps is not None:
        raise ValueError("choose bounded-support Hoeffding OR variance-cap Cantelli")
    if penalty_cap is not None:
        cap = float(penalty_cap)
        if not np.isfinite(cap) or cap <= 0 or any(np.any(v > cap + 1e-12)
                                                  for v in arrays.values()):
            raise ValueError("penalty cap must be finite, positive, and cover observations")
        radius = cap * np.sqrt(np.log(1 / alpha) / 2 * sum(
            weights[g] ** 2 / len(v) for g, v in arrays.items()))
        upper = min(cap, mean + radius)
        method = "weighted_hoeffding"
        assumptions = ["tail penalty is almost surely bounded by the declared cap"]
    elif variance_caps is not None:
        if any(g not in variance_caps or not np.isfinite(variance_caps[g])
               or variance_caps[g] < 0 for g in arrays):
            raise ValueError("each positive-weight group needs a finite nonnegative variance cap")
        variance = sum(weights[g] ** 2 * variance_caps[g] / len(v)
                       for g, v in arrays.items())
        radius = np.sqrt(variance * (1 - alpha) / alpha)
        upper = mean + radius
        method = "one_sided_cantelli_known_variance"
        assumptions = ["each declared cap upper-bounds population variance of its tail penalty"]
    else:
        return {"status": "refused", "reason":
                "unbounded tail magnitude requires an external penalty or variance cap"}
    return {"status": "conditional", "tail_penalty_upper": float(upper),
            "tail_penalty_mean": float(mean), "radius": float(radius),
            "method": method, "alpha": float(alpha), "loss": loss,
            "assumptions": assumptions + [
                "iid calibration outcomes within groups",
                "tail laws and target weights are stable from calibration to target",
                "clip, loss, weights, and tail method fixed without calibration outcomes",
            ]}


def transfer_interval(clipped: dict, tail: dict) -> dict:
    if clipped.get("status") != "conditional" or tail.get("status") != "conditional":
        return {"status": "refused", "reason": "both clipped interval and tail cap must issue"}
    if clipped["loss"] != tail["loss"]:
        raise ValueError("loss mismatch")
    lower = float(clipped["lower"])
    upper = float(clipped["upper"] + tail["tail_penalty_upper"])
    if clipped["loss"].startswith("W"):
        upper = min(1., upper)
    out = {"status": "conditional", "claim_tier": "conditional",
           "loss": clipped["loss"], "lower": lower, "upper": upper,
           "clipped_upper": float(clipped["upper"]),
           "tail_penalty_upper": float(tail["tail_penalty_upper"]),
           "target": "original-outcome Bayes risk for declared group representation",
           "raw_information_certificate": False,
           "alpha_components": [float(clipped["alpha_endpoint"]), float(tail["alpha"])],
           "assumptions": clipped.get("assumptions", []) + tail.get("assumptions", [])}
    if clipped["loss"].startswith("W"):
        out["miss_probability_interval"] = [lower, upper]
        out["within_window_percentage_interval"] = [100 * (1 - upper),
                                                     100 * (1 - lower)]
    return out
