"""Independent bounded-outcome calibration of group-law W1 caps.

Only valid when each declared group has one common conditional outcome law,
calibration outcomes are iid within group and independent of audit outcomes,
and all cap/model/clip choices are frozen before audit outcomes are observed.
No predictive model is trained. This is not a continuous-covariate law cap.
"""
from __future__ import annotations

from typing import Hashable

import numpy as np
from scipy.stats import wasserstein_distance


def calibrate_group_w1(
    samples: dict[Hashable, np.ndarray], *,
    clip: tuple[float, float], delta: float = 0.025,
) -> dict:
    """Simultaneous pairwise W1 upper caps via DKW and a union bound.

    For G groups and confidence 1-delta, DKW gives each empirical CDF
    sup-error <= sqrt(log(2G/delta)/(2n_g)). On [lo,hi], integrate the two
    CDF errors to bound W1(P_g,P_h) by empirical W1 plus
    (hi-lo)*(eps_g+eps_h). Same-group distance is identically zero by the
    group-law assumption; it needs no calibration penalty.
    """
    if not 0 < delta < 1 or not samples:
        raise ValueError("nonempty samples and delta in (0,1) required")
    lo, hi = map(float, clip)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("finite ordered clip required")
    group_names = list(samples)
    if len(group_names) < 2:
        raise ValueError("at least two calibrated groups required")
    arrays = {}
    eps = {}
    for group in group_names:
        a = np.asarray(samples[group], float).ravel()
        if len(a) < 2 or not np.isfinite(a).all():
            raise ValueError(f"group {group!r} needs >=2 finite calibration outcomes")
        arrays[group] = np.clip(a, lo, hi)
        eps[group] = float(np.sqrt(np.log(2 * len(samples) / delta) / (2 * len(a))))
    caps = {}
    raw = {}
    for i, group in enumerate(group_names):
        caps[(group, group)] = 0.0
        raw[(group, group)] = 0.0
        for other in group_names[i + 1:]:
            empirical = float(wasserstein_distance(arrays[group], arrays[other]))
            upper = min(hi - lo, empirical + (hi - lo) * (eps[group] + eps[other]))
            caps[(group, other)] = caps[(other, group)] = upper
            raw[(group, other)] = raw[(other, group)] = empirical
    return {"caps": caps, "empirical_w1": raw, "cdf_radii": eps,
            "group_sizes": {g: len(arrays[g]) for g in group_names},
            "clip": (lo, hi), "delta": float(delta),
            "assumptions": ["declared group-homogeneous clipped conditional laws",
                            "independent iid calibration outcomes per group",
                            "clip and groups fixed without audit outcomes"]}


def mean_ordered_block_cap(groups: np.ndarray, blocks: np.ndarray,
                           calibration: dict) -> float:
    """Mean cap over all ordered within-block group-law pairs, diagonal included."""
    labels = np.asarray(groups, dtype=object).ravel()
    idx = np.asarray(blocks, int)
    if (idx.ndim != 2 or idx.shape[0] < 2 or idx.shape[1] < 2
            or np.any(idx < 0) or np.any(idx >= len(labels))
            or np.unique(idx).size != idx.size):
        raise ValueError("blocks must be valid and globally disjoint")
    caps = calibration["caps"]
    total = 0.0
    for block in idx:
        for i in block:
            for j in block:
                pair = (labels[i], labels[j])
                if pair not in caps:
                    raise ValueError(f"uncalibrated group pair {pair!r}")
                total += caps[pair]
    return total / (idx.shape[0] * idx.shape[1] ** 2)


def calibrate_mean_block_w1_direct(
    samples: dict[Hashable, np.ndarray], groups: np.ndarray, blocks: np.ndarray,
    *, clip: tuple[float, float], delta: float = 0.025,
) -> dict:
    """One-sided cap on the *average* ordered W1, without pairwise union cost.

    The pair weights are frozen without calibration or audit outcomes. The
    empirical weighted average A of pairwise W1 is upward biased for its
    population counterpart by joint convexity of W1. Changing one bounded
    calibration observation in group g changes A by at most D*d_g/n_g,
    where d_g is the sum of incident (ordered-pair) weights. McDiarmid then
    gives E[A] <= A + D*sqrt(log(1/delta)/2 * sum_g d_g^2/n_g).
    Combining population <= E[A] yields the advertised upper cap.
    """
    if not 0 < delta < 1 or not samples:
        raise ValueError("nonempty samples and delta in (0,1) required")
    lo, hi = map(float, clip)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("finite ordered clip required")
    labels = np.asarray(groups, dtype=object).ravel()
    idx = np.asarray(blocks, int)
    if (idx.ndim != 2 or idx.shape[0] < 2 or idx.shape[1] < 2
            or np.any(idx < 0) or np.any(idx >= len(labels))
            or np.unique(idx).size != idx.size):
        raise ValueError("blocks must be valid and globally disjoint")
    arrays = {}
    for group, sample in samples.items():
        a = np.asarray(sample, float).ravel()
        if len(a) < 2 or not np.isfinite(a).all():
            raise ValueError(f"group {group!r} needs >=2 finite calibration outcomes")
        arrays[group] = np.clip(a, lo, hi)
    pair_counts = {}
    pair_labels = {}
    for block in idx:
        for i in block:
            for j in block:
                a, b = labels[i], labels[j]
                if a not in arrays or b not in arrays:
                    raise ValueError(f"uncalibrated group {a!r} or {b!r}")
                if a == b:
                    continue
                key = frozenset((a, b))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                pair_labels[key] = (a, b)
    denominator = idx.shape[0] * idx.shape[1] ** 2
    width = hi - lo
    degrees = {g: 0.0 for g in arrays}
    empirical = 0.0
    pairwise_dkw_cap = 0.0
    cross_weight = 0.0
    eps = {g: np.sqrt(np.log(2 * len(arrays) / delta) / (2 * len(a)))
           for g, a in arrays.items()}
    for key, count in pair_counts.items():
        a, b = pair_labels[key]
        weight = count / denominator
        pair_empirical = float(wasserstein_distance(arrays[a], arrays[b]))
        empirical += weight * pair_empirical
        pairwise_dkw_cap += weight * min(
            width, pair_empirical + width * (eps[a] + eps[b]))
        degrees[a] += weight
        degrees[b] += weight
        cross_weight += weight
    radius = width * np.sqrt(np.log(1 / delta) / 2 * sum(
        degree ** 2 / len(arrays[g]) for g, degree in degrees.items()))
    cap = min(width * cross_weight, empirical + radius)
    return {"mean_w1_cap": float(cap), "empirical_mean_w1": float(empirical),
            "pairwise_dkw_mean_cap": float(pairwise_dkw_cap),
            "concentration_radius": float(radius),
            "cross_group_pair_weight": float(cross_weight),
            "n_distinct_cross_pairs": len(pair_counts),
            "calibration_group_sizes": {g: len(a) for g, a in arrays.items()},
            "clip": (lo, hi), "delta": float(delta),
            "assumptions": ["block/group/clip/weight design fixed without calibration or audit outcomes",
                            "common clipped conditional law within each group",
                            "independent iid calibration outcomes per group",
                            "calibration outcomes independent of audit outcomes"]}
