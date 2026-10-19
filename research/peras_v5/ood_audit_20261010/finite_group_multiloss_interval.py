"""Training-free SAA/held-out risk intervals for fixed finite groups.

These are conditional intervals for CLIPPED outcomes and the declared group
representation only. They are not raw-text/full-feature sufficiency claims.
"""
from __future__ import annotations

import numpy as np

from core import erm_action, erm_risk
from peras.core.concentration import bentkus_lower_mean

LOSSES = ("MAE", "MSE", "Q10", "Q50", "Q90", "W15", "W30", "W60")
SCORE_CAPS = {"MAE": 30., "MSE": 1000., "Q10": 6., "Q50": 15.,
              "Q90": 6., "W15": 1., "W30": 1., "W60": 1.}


def empirical_floor_all(y: np.ndarray, loss: str) -> float:
    if loss.startswith("W"):
        s = np.sort(y)
        h = float(loss[1:])
        left = best = 0
        for right in range(len(s)):
            while s[right] - s[left] > 2 * h:
                left += 1
            best = max(best, right - left + 1)
        return float(1 - best / len(s))
    if loss == "MAE":
        return float(np.mean(np.abs(y - np.median(y))))
    if loss == "MSE":
        return float(np.var(y))
    q = float(loss[1:]) / 100
    a = np.quantile(y, q, method="inverted_cdf")
    u = y - a
    return float(np.mean(np.maximum(q * u, (q - 1) * u)))


def block_statistics(t: np.ndarray, loss: str) -> tuple[np.ndarray, np.ndarray]:
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("quantile must lie strictly inside (0,1)")
        a = np.quantile(t, q, axis=1, method="inverted_cdf")
        u = t - a[:, None]
        saa = np.maximum(q * u, (q - 1) * u).mean(axis=1)
        held_a = np.quantile(t[:, 1:], q, axis=1, method="inverted_cdf")
        held_u = t[:, 0] - held_a
        held = np.maximum(q * held_u, (q - 1) * held_u)
        return saa, held
    if loss not in ("MAE", "MSE") and not loss.startswith("W"):
        raise ValueError("unsupported loss")
    saa = erm_risk(loss, t)
    a = erm_action(loss, t[:, 1:])
    u = t[:, 0] - a
    if loss == "MAE":
        held = np.abs(u)
    elif loss == "MSE":
        held = u ** 2
    else:
        h = float(loss[1:])
        if h <= 0:
            raise ValueError("window must be positive")
        held = (np.abs(u) > h).astype(float)
    return saa, held


def interval(samples: dict, weights: dict, *, loss: str,
             clip: tuple[float, float] = (0., 300.), k: int = 16,
             score_cap: float | None = None, alpha_endpoint: float = .05 / 16) -> dict:
    lo, hi = map(float, clip)
    if (not np.isfinite([lo, hi]).all() or lo >= hi or k < 2
            or not 0 < alpha_endpoint < 1):
        raise ValueError("valid clip, tuple size, alpha required")
    if (not weights or any(not np.isfinite(w) or w < 0 for w in weights.values())
            or not np.isclose(sum(weights.values()), 1., atol=1e-10)):
        raise ValueError("weights must be finite, nonnegative, and sum to one")
    width = hi - lo
    if loss == "MAE":
        upper_range = width
    elif loss == "MSE":
        upper_range = width ** 2
    elif loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        upper_range = max(q, 1 - q) * width
    elif loss.startswith("W") and float(loss[1:]) > 0:
        upper_range = 1.
    else:
        return {"status": "refused", "reason": "unsupported loss"}
    cap = upper_range if score_cap is None else float(score_cap)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("positive finite score cap required")
    cap = min(cap, upper_range)
    lower_center = upper_center = sum_squares = plugin = 0.
    blocks = 0
    for group, weight in weights.items():
        if weight == 0:
            continue
        if group not in samples:
            return {"status": "refused", "reason": "missing positive-weight calibration group"}
        y = np.asarray(samples[group], float).ravel()
        if len(y) < k or not np.isfinite(y).all():
            return {"status": "refused", "reason": "undersized or nonfinite calibration group"}
        y = np.clip(y, lo, hi)
        plugin += weight * empirical_floor_all(y, loss)
        m = len(y) // k
        saa, held = block_statistics(y[:m * k].reshape(m, k), loss)
        lower_center += weight * float(np.minimum(saa, cap).mean())
        # Upper scores must NOT be winsorized below their true loss range.
        upper_center += weight * float(held.mean())
        sum_squares += weight ** 2 / m
        blocks += m
    factor = np.sqrt(np.log(1 / alpha_endpoint) / 2 * sum_squares)
    lower = max(0., lower_center - cap * factor)
    upper = min(upper_range, upper_center + upper_range * factor)
    if lower > upper + 1e-12:
        return {"status": "refused", "reason": "contradictory interval endpoints"}
    out = {"status": "conditional", "claim_tier": "conditional", "loss": loss,
            "lower": float(lower), "upper": float(upper),
            "lower_center": float(lower_center), "upper_center": float(upper_center),
            "plugin_clipped": float(plugin),
            "plugin_claim_tier": "exploratory point estimate, not interval endpoint",
            "n_blocks": blocks, "k": k, "alpha_endpoint": alpha_endpoint,
            "outcome_clip": [lo, hi], "lower_score_cap": cap,
            "upper_score_range": upper_range,
            "target": "audit-weighted clipped-outcome Bayes risk for declared group representation",
            "raw_information_certificate": False,
            "original_outcome_upper_certified": False,
            "assumptions": ["iid calibration scenarios within each group-reference law",
                            "reference laws stable from calibration to target",
                            "weights, clips, grouping, k fixed without calibration outcomes"]}
    if loss.startswith("W"):
        out["miss_probability_interval"] = [float(lower), float(upper)]
        out["within_window_percentage_interval"] = [float(100 * (1 - upper)),
                                                     float(100 * (1 - lower))]
    return out


def partial_group_lower(samples: dict, weights: dict, *, loss: str,
                        clip: tuple[float, float] = (0., 300.), k: int = 16,
                        score_cap: float | None = None,
                        alpha_endpoint: float = .05) -> dict:
    """Lower-bound a fine representation while safely omitting sparse cells.

    Each target group with at least k calibration outcomes contributes disjoint
    SAA blocks. Missing/undersized groups contribute the universally valid zero
    lower bound. Inclusion depends on counts only, never on outcomes.
    """
    lo, hi = map(float, clip)
    if (not np.isfinite([lo, hi]).all() or lo >= hi or k < 2
            or not 0 < alpha_endpoint < 1):
        raise ValueError("valid clip, tuple size, alpha required")
    if (not weights or any(not np.isfinite(w) or w < 0 for w in weights.values())
            or not np.isclose(sum(weights.values()), 1., atol=1e-10)):
        raise ValueError("weights must be finite, nonnegative, and sum to one")
    width = hi - lo
    if loss == "MAE":
        loss_range = width
    elif loss == "MSE":
        loss_range = width ** 2
    elif loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        loss_range = max(q, 1 - q) * width
    elif loss.startswith("W") and float(loss[1:]) > 0:
        loss_range = 1.
    else:
        return {"status": "refused", "reason": "unsupported loss"}
    cap = loss_range if score_cap is None else min(float(score_cap), loss_range)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("positive finite score cap required")
    center = sum_squares = included_mass = 0.
    blocks = included_groups = 0
    for group, weight in weights.items():
        if weight == 0 or group not in samples:
            continue
        y = np.asarray(samples[group], float).ravel()
        if not np.isfinite(y).all():
            return {"status": "refused", "reason": "nonfinite calibration outcome"}
        m = len(y) // k
        if m < 1:
            continue
        y = np.clip(y[:m * k], lo, hi).reshape(m, k)
        saa, _ = block_statistics(y, loss)
        center += weight * float(np.minimum(saa, cap).mean())
        sum_squares += weight ** 2 / m
        included_mass += weight
        blocks += m
        included_groups += 1
    radius = cap * np.sqrt(np.log(1 / alpha_endpoint) / 2 * sum_squares)
    lower = max(0., center - radius)
    return {
        "status": "conditional", "claim_tier": "conditional", "loss": loss,
        "lower": float(lower), "lower_center": float(center),
        "radius": float(radius), "n_blocks": int(blocks), "k": int(k),
        "included_groups": int(included_groups),
        "total_target_groups": int(sum(w > 0 for w in weights.values())),
        "included_target_mass": float(included_mass),
        "omitted_target_mass": float(max(0., 1 - included_mass)),
        "alpha_endpoint": float(alpha_endpoint), "outcome_clip": [lo, hi],
        "lower_score_cap": float(cap),
        "target": "audit-weighted clipped Bayes risk for the full declared fine representation",
        "raw_information_certificate": False,
        "assumptions": [
            "independent calibration outcomes within included fine groups",
            "fine-group conditional laws are stable from calibration to target",
            "weights, group map, clip, k, cap, and count-only inclusion were fixed without calibration outcomes",
            "omitted target groups contribute the valid zero lower bound",
        ],
    }


def partial_group_portfolio(samples: dict, weights: dict, *, loss: str,
                            ks: tuple[int, ...] = (2, 4, 8, 16),
                            clip: tuple[float, float] = (0., 300.),
                            score_cap: float | None = None,
                            alpha_family: float = .05) -> dict:
    """Selection-safe maximum over predeclared partial-cell block sizes."""
    if (not ks or len(set(ks)) != len(ks)
            or any(not isinstance(k, (int, np.integer)) or k < 2 for k in ks)
            or not 0 < alpha_family < 1):
        raise ValueError("unique block sizes >=2 and valid family alpha required")
    per_route = alpha_family / len(ks)
    candidates = [partial_group_lower(
        samples, weights, loss=loss, clip=clip, k=int(k),
        score_cap=score_cap, alpha_endpoint=per_route) for k in ks]
    if any(row.get("status") != "conditional" for row in candidates):
        return {"status": "refused", "reason": "a portfolio route refused"}
    chosen = max(candidates, key=lambda row: row["lower"])
    return {
        **chosen,
        "selected_k": chosen["k"],
        "candidate_ks": [int(k) for k in ks],
        "alpha_family": float(alpha_family),
        "alpha_per_route": float(per_route),
        "selection": "maximum of jointly Bonferroni-covered lower endpoints",
        "candidates": [{
            "k": row["k"], "lower": row["lower"],
            "included_target_mass": row["included_target_mass"],
            "omitted_target_mass": row["omitted_target_mass"],
            "n_blocks": row["n_blocks"],
        } for row in candidates],
    }


def balanced_bentkus_interval(samples: dict, target_counts: dict, *, loss: str,
                              clip: tuple[float, float] = (0., 300.), k: int = 16,
                              score_cap: float | None = None,
                              alpha_endpoint: float = .05 / 16) -> dict:
    """Bentkus interval with exact finite-target group proportions.

    If target group g occurs c_g times, retain r*c_g disjoint calibration
    blocks for the largest common integer r. Their equal-weight mean has
    expectation equal to the target-count-weighted group mean. Bentkus allows
    independent nonidentical bounded scores. Surplus blocks are discarded.
    """
    lo, hi = map(float, clip)
    if (not np.isfinite([lo, hi]).all() or lo >= hi or k < 2
            or not 0 < alpha_endpoint < 1 or not target_counts
            or any(not isinstance(c, (int, np.integer)) or c <= 0
                   for c in target_counts.values())):
        raise ValueError("valid clip, k, alpha, and positive integer target counts required")
    width = hi - lo
    if loss == "MAE":
        upper_range = width
    elif loss == "MSE":
        upper_range = width ** 2
    elif loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        upper_range = max(q, 1 - q) * width
    elif loss.startswith("W") and float(loss[1:]) > 0:
        upper_range = 1.
    else:
        return {"status": "refused", "reason": "unsupported loss"}
    cap = upper_range if score_cap is None else min(float(score_cap), upper_range)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("positive finite score cap required")
    arrays = {}
    possible = []
    for group, count in target_counts.items():
        if group not in samples:
            return {"status": "refused", "reason": "missing target group in calibration"}
        y = np.asarray(samples[group], float).ravel()
        if not np.isfinite(y).all():
            return {"status": "refused", "reason": "nonfinite calibration outcome"}
        arrays[group] = np.clip(y, lo, hi)
        possible.append((len(y) // k) // count)
    multiplier = min(possible)
    if multiplier < 1:
        return {"status": "refused", "reason": "insufficient blocks for exact target proportions"}
    lower_scores, upper_scores = [], []
    used_outcomes = 0
    for group, count in target_counts.items():
        blocks = multiplier * count
        y = arrays[group][:blocks * k].reshape(blocks, k)
        saa, held = block_statistics(y, loss)
        lower_scores.append(np.minimum(saa, cap))
        upper_scores.append(held)
        used_outcomes += blocks * k
    low_values = np.concatenate(lower_scores)
    high_values = np.concatenate(upper_scores)
    lower = bentkus_lower_mean(low_values, cap, alpha_endpoint)
    # Transform H in [0,R] into R-H. A lower confidence bound on its average
    # expectation is an upper confidence bound on the average expectation of H.
    upper = upper_range - bentkus_lower_mean(
        upper_range - high_values, upper_range, alpha_endpoint)
    if lower > upper + 1e-12:
        return {"status": "refused", "reason": "contradictory Bentkus endpoints"}
    out = {"status": "conditional", "claim_tier": "conditional", "loss": loss,
           "lower": float(lower), "upper": float(upper),
           "lower_center": float(low_values.mean()),
           "upper_center": float(high_values.mean()),
           "n_blocks": int(len(low_values)), "k": int(k),
           "balance_multiplier": int(multiplier),
           "used_calibration_outcomes": int(used_outcomes),
           "available_calibration_outcomes": int(sum(len(v) for v in arrays.values())),
           "alpha_endpoint": float(alpha_endpoint), "outcome_clip": [lo, hi],
           "lower_score_cap": float(cap), "upper_score_range": float(upper_range),
           "target": "finite-target-count-weighted clipped group Bayes risk",
           "raw_information_certificate": False,
           "original_outcome_upper_certified": False,
           "assumptions": ["independent calibration outcomes within and across groups",
                           "group-reference laws stable from calibration to target",
                           "target group counts, order, clips, k, and caps fixed without calibration outcomes",
                           "discarded surplus blocks are ignored without outcome inspection"]}
    if loss.startswith("W"):
        out["miss_probability_interval"] = [float(lower), float(upper)]
        out["within_window_percentage_interval"] = [float(100 * (1 - upper)),
                                                     float(100 * (1 - lower))]
    return out


def randomized_mixture_bentkus_interval(
    samples: dict, target_counts: dict, *, loss: str,
    randomization_seed: int, n_blocks: int = 200,
    clip: tuple[float, float] = (0., 300.), k: int = 16,
    score_cap: float | None = None, alpha_endpoint: float = .05 / 16,
) -> dict:
    """Bentkus interval for a pre-randomized audit-mixture block sample.

    Group labels are sampled iid with probabilities equal to finite audit
    proportions, independently of all outcomes. Each occurrence consumes a
    fresh disjoint calibration block from that group. The claim is marginal
    over this registered randomization and the calibration outcomes. If any
    requested group lacks capacity, the function refuses rather than redraws.
    """
    if (not isinstance(randomization_seed, (int, np.integer)) or n_blocks < 2
            or not target_counts or any(not isinstance(c, (int, np.integer)) or c <= 0
                                        for c in target_counts.values())):
        raise ValueError("registered integer seed, >=2 blocks, and target counts required")
    lo, hi = map(float, clip)
    if not np.isfinite([lo, hi]).all() or lo >= hi or k < 2 or not 0 < alpha_endpoint < 1:
        raise ValueError("valid clip, k, and alpha required")
    width = hi - lo
    if loss == "MAE":
        upper_range = width
    elif loss == "MSE":
        upper_range = width ** 2
    elif loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        upper_range = max(q, 1 - q) * width
    elif loss.startswith("W") and float(loss[1:]) > 0:
        upper_range = 1.
    else:
        return {"status": "refused", "reason": "unsupported loss"}
    cap = upper_range if score_cap is None else min(float(score_cap), upper_range)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("positive finite score cap required")
    labels = list(target_counts)
    counts = np.array([target_counts[g] for g in labels], float)
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(int(randomization_seed))
    schedule_index = rng.choice(len(labels), size=n_blocks, p=probabilities)
    requested = np.bincount(schedule_index, minlength=len(labels))
    arrays = {}
    for j, group in enumerate(labels):
        if group not in samples:
            return {"status": "refused", "reason": "scheduled group missing from calibration"}
        y = np.asarray(samples[group], float).ravel()
        if not np.isfinite(y).all():
            return {"status": "refused", "reason": "nonfinite calibration outcome"}
        if len(y) // k < requested[j]:
            return {"status": "refused", "reason": "registered random schedule exceeds group capacity"}
        arrays[group] = np.clip(y, lo, hi)
    positions = {group: 0 for group in labels}
    lower_scores, upper_scores = [], []
    for j in schedule_index:
        group = labels[int(j)]
        start = positions[group] * k
        positions[group] += 1
        block = arrays[group][start:start + k][None, :]
        saa, held = block_statistics(block, loss)
        lower_scores.append(min(float(saa[0]), cap))
        upper_scores.append(float(held[0]))
    low_values = np.asarray(lower_scores)
    high_values = np.asarray(upper_scores)
    lower = bentkus_lower_mean(low_values, cap, alpha_endpoint)
    upper = upper_range - bentkus_lower_mean(
        upper_range - high_values, upper_range, alpha_endpoint)
    if lower > upper + 1e-12:
        return {"status": "refused", "reason": "contradictory randomized endpoints"}
    out = {"status": "conditional", "claim_tier": "conditional", "loss": loss,
           "lower": float(lower), "upper": float(upper),
           "lower_center": float(low_values.mean()), "upper_center": float(high_values.mean()),
           "n_blocks": int(n_blocks), "k": int(k),
           "randomization_seed": int(randomization_seed),
           "scheduled_group_counts": {str(labels[j]): int(requested[j])
                                      for j in range(len(labels))},
           "alpha_endpoint": float(alpha_endpoint), "outcome_clip": [lo, hi],
           "lower_score_cap": float(cap), "upper_score_range": float(upper_range),
           "target": "audit-mixture-weighted clipped group Bayes risk",
           "coverage_scope": "marginal over registered group randomization and calibration outcomes",
           "raw_information_certificate": False,
           "original_outcome_upper_certified": False,
           "assumptions": ["registered group schedule is iid from frozen audit proportions and independent of outcomes",
                           "each scheduled score uses a fresh disjoint calibration block",
                           "group-reference laws stable from calibration to target",
                           "no redraw, seed search, or schedule adaptation after capacity/outcome inspection"]}
    if loss.startswith("W"):
        out["miss_probability_interval"] = [float(lower), float(upper)]
        out["within_window_percentage_interval"] = [float(100 * (1 - upper)),
                                                     float(100 * (1 - lower))]
    return out
