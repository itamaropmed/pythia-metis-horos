"""Selection-safe train-free MAE witness across paired and grouped families."""
from __future__ import annotations

import numpy as np

from pythia.core.group_witness import grouped_decision_envelope
from pythia.core.replicates import (
    paired_replicate_clipped_envelope, paired_replicate_lower_bound,
)


def mae_pair_group_envelope(
    representation: np.ndarray,
    outcomes: np.ndarray,
    entities: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    groupings: dict[int, np.ndarray],
    *,
    lipschitz_bound: float,
    outcome_term_clips: tuple[float, ...],
    alpha: float = .05,
    methods: tuple[str, ...] = (
        "conditional_hoeffding", "conditional_empirical_bernstein",
    ),
) -> dict:
    """Bound the same fixed-design Bayes MAE by a simultaneous maximum.

    Pairing, grouping, metric, clips, methods, and L must be fixed without
    evaluation outcomes. Distinct entities must have conditionally independent
    outcomes. Bonferroni covers both witness families and their inner choices.
    """
    x = np.asarray(representation, float)
    y = np.asarray(outcomes, float).ravel()
    entity = np.asarray(entities).astype(str).ravel()
    left = np.asarray(pair_left)
    right = np.asarray(pair_right)
    if (x.ndim != 2 or x.shape[0] != y.size or entity.size != y.size
            or left.ndim != 1 or right.ndim != 1 or left.size != right.size
            or left.size * 2 != y.size or left.dtype.kind not in "iu"
            or right.dtype.kind not in "iu"):
        raise ValueError("pair and outcome arrays must cover the same even-sized cohort")
    if not np.array_equal(np.sort(np.concatenate([left, right])), np.arange(y.size)):
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "paired family does not partition the grouped cohort"}
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("representation must be finite and in [0,1]")
    if len(np.unique(entity)) != len(entity):
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "sampling entities must be distinct"}
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("nonempty, unique method registry required")
    if any(method not in {"conditional_hoeffding", "conditional_empirical_bernstein"}
           for method in methods):
        raise ValueError("only conditional independent-entity methods are supported")
    distance = np.mean(np.abs(x[left] - x[right]), axis=1)
    pairs = []
    for method in methods:
        row = paired_replicate_clipped_envelope(
            y[left], y[right], distance, entity[left], entity[right],
            np.arange(left.size), loss="mae", lipschitz_bound=lipschitz_bound,
            distance_bound=1., outcome_difference_bound=None,
            outcome_term_clips=outcome_term_clips,
            alpha=alpha / (2 * len(methods)), concentration=method,
        )
        if row["status"] != "complete":
            return row
        pairs.append(row)
    grouped = grouped_decision_envelope(
        x, y, entity, groupings, loss="mae", lipschitz_bound=lipschitz_bound,
        outcome_range_bound=None, outcome_term_clips=outcome_term_clips,
        alpha=alpha / 2, methods=methods,
    )
    if grouped["status"] != "complete":
        return grouped
    candidates = [("paired", row) for row in pairs] + [("grouped", grouped)]
    family, selected = max(candidates, key=lambda item: item[1]["bound"])
    result = dict(selected)
    result.update({
        "selected_family": family,
        "bound": float(selected["bound"]),
        "diagnostic_status": (
            "uninformative_at_declared_metric" if selected["bound"] <= 0
            else "positive_lower_witness"
        ),
        "confidence": float(1 - alpha),
        "simultaneous_across_families": True,
        "alpha_allocation_across_families": {"paired": alpha / 2, "grouped": alpha / 2},
        "family_bounds": {
            "paired": float(max(row["bound"] for row in pairs)),
            "grouped": float(grouped["bound"]),
        },
        "estimand": "average conditional Bayes MAE over every audited entity",
        "registry_warning": (
            "the metric, L, pair/group constructions, clip and method registries "
            "must be fixed without evaluation outcomes; conditional entity "
            "independence and the median Lipschitz cap require external audit"
        ),
        "point_estimate_is_certified": False,
    })
    return result


def decision_pair_group_envelope(
    representation: np.ndarray,
    outcomes: np.ndarray,
    entities: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    groupings: dict[int, np.ndarray],
    *,
    loss: str,
    lipschitz_bound: float,
    quantile: float | None = None,
    window: float | None = None,
    outcome_range_bound: float | None = None,
    outcome_term_clips: tuple[float, ...] | None = None,
    alpha: float = .05,
    family_policy: str = "adaptive",
    group_center_rule: str = "mean",
    group_witness_rule: str = "radial",
    group_pinball_range_rule: str = "asymptotic",
    methods: tuple[str, ...] = (
        "conditional_hoeffding", "conditional_empirical_bernstein",
        "conditional_bentkus",
    ),
) -> dict:
    """Simultaneous conditional pinball/window-miss Bayes-risk lower bound.

    ``family_policy`` may be ``paired``, ``grouped``, or ``adaptive``. A policy
    fixed without evaluation outcomes spends no alpha on an unused family;
    adaptive selection covers both families. Multiple losses, windows,
    quantiles, slices, or representations need outer multiplicity allocation.
    """
    selected_loss = loss.lower()
    if selected_loss not in {"pinball", "window_miss"}:
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "joint decision envelope supports pinball and window_miss only"}
    x = np.asarray(representation, float)
    y = np.asarray(outcomes, float).ravel()
    entity = np.asarray(entities).astype(str).ravel()
    left = np.asarray(pair_left)
    right = np.asarray(pair_right)
    if (x.ndim != 2 or x.shape[0] != y.size or entity.size != y.size
            or left.ndim != 1 or right.ndim != 1 or left.size != right.size
            or left.size * 2 != y.size or left.dtype.kind not in "iu"
            or right.dtype.kind not in "iu"):
        raise ValueError("pairs and groups must cover the same even-sized cohort")
    if not np.array_equal(np.sort(np.concatenate([left, right])), np.arange(y.size)):
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "paired family does not partition the grouped cohort"}
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("representation must be finite and in [0,1]")
    if len(np.unique(entity)) != len(entity):
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "sampling entities must be distinct"}
    if not 0 < alpha < 1 or not methods or len(set(methods)) != len(methods):
        raise ValueError("valid alpha and nonempty unique method registry required")
    if family_policy not in {"paired", "grouped", "adaptive"}:
        raise ValueError("family_policy must be paired, grouped, or adaptive")
    allowed = {"conditional_hoeffding", "conditional_empirical_bernstein",
               "conditional_bentkus", "conditional_entity_mcd"}
    if any(method not in allowed for method in methods):
        raise ValueError("only independent-entity conditional methods are supported")
    if (family_policy != "grouped" and "conditional_entity_mcd" in methods):
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "entity bounded differences currently supports grouped-only policy"}
    if selected_loss == "window_miss" and outcome_term_clips is not None:
        raise ValueError("window miss is naturally bounded; clips are unsupported")
    if selected_loss == "pinball" and outcome_range_bound is None and not outcome_term_clips:
        raise ValueError("pinball needs a predeclared outcome range or clip registry")
    clips = (None,) if outcome_term_clips is None else tuple(outcome_term_clips)
    if not clips or (outcome_term_clips is not None and (
            any(not np.isfinite(clip) or clip <= 0 for clip in clips)
            or any(b <= a for a, b in zip(clips, clips[1:])))):
        raise ValueError("clips must be finite, positive, and strictly increasing")
    distance = np.mean(np.abs(x[left] - x[right]), axis=1)
    family_alpha = alpha / 2 if family_policy == "adaptive" else alpha
    candidates = []
    if family_policy in {"paired", "adaptive"}:
        paired_rows = []
        for method in methods:
            for clip in clips:
                row = paired_replicate_lower_bound(
                    y[left], y[right], distance, entity[left], entity[right],
                    np.arange(left.size), loss=selected_loss,
                    lipschitz_bound=lipschitz_bound, distance_bound=1.,
                    outcome_difference_bound=outcome_range_bound,
                    outcome_term_clip=clip, quantile=quantile, window=window,
                    alpha=family_alpha / (len(methods) * len(clips)),
                    concentration=method,
                )
                if row["status"] != "complete":
                    return row
                paired_rows.append(row)
        candidates.append(("paired", max(paired_rows, key=lambda row: row["bound"])))
    if family_policy in {"grouped", "adaptive"}:
        grouped = grouped_decision_envelope(
            x, y, entity, groupings, loss=selected_loss,
            lipschitz_bound=lipschitz_bound,
            outcome_range_bound=outcome_range_bound,
            outcome_term_clips=outcome_term_clips,
            quantile=quantile, window=window, alpha=family_alpha, methods=methods,
            center_rule=group_center_rule,
            witness_rule=group_witness_rule,
            pinball_range_rule=group_pinball_range_rule,
        )
        if grouped["status"] != "complete":
            return grouped
        candidates.append(("grouped", grouped))
    family, selected = max(candidates, key=lambda item: item[1]["bound"])
    result = dict(selected)
    result.update({
        "selected_family": family,
        "family_policy": family_policy,
        "group_center_rule": group_center_rule,
        "group_witness_rule": group_witness_rule,
        "group_pinball_range_rule": group_pinball_range_rule,
        "family_bounds": {name: row["bound"] for name, row in candidates},
        "confidence": float(1 - alpha),
        "simultaneous_across_families": family_policy == "adaptive",
        "alpha_allocation_across_families": {
            "paired": family_alpha if family_policy in {"paired", "adaptive"} else 0.,
            "grouped": family_alpha if family_policy in {"grouped", "adaptive"} else 0.,
        },
        "diagnostic_status": (
            "uninformative_at_declared_metric" if selected["bound"] <= 0
            else "positive_lower_witness"
        ),
        "point_estimate_is_certified": False,
        "outer_multiplicity_required": (
            "allocate alpha across every predeclared loss, window, quantile, "
            "representation, slice, and period"
        ),
        "registry_warning": (
            "conditional action smoothness, independent entities, outcome-blind "
            "metric/design/family policy, and range/clip provenance require external audit"
        ),
    })
    if selected_loss == "window_miss":
        result["attainable_within_window_percentage_upper"] = float(
            100 * (1 - result["bound"])
        )
    return result


def geometry_routed_decision_envelope(
    representation: np.ndarray,
    outcomes: np.ndarray,
    entities: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    groupings: dict[int, np.ndarray],
    *,
    loss: str,
    lipschitz_bound: float,
    quantile: float | None = None,
    window: float | None = None,
    outcome_range_bound: float | None = None,
    alpha: float = .05,
    pinball_ratio_threshold: float = .01,
    window_ratio_threshold: float = .5,
) -> dict:
    """Route by declared covariate geometry, never by evaluation outcomes.

    Thresholds are development heuristics, not population-optimal constants.
    A grouped pinball route remains exploratory until exact-LP arithmetic is
    independently audited; supported window and paired routes are conditional
    on the same action-smoothness and independence premises as their witnesses.
    """
    selected_loss = loss.lower()
    if selected_loss not in {"pinball", "window_miss"}:
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "geometry route supports pinball and window_miss only"}
    x = np.asarray(representation, float)
    left, right = np.asarray(pair_left), np.asarray(pair_right)
    if (x.ndim != 2 or left.ndim != 1 or right.ndim != 1
            or left.size != right.size or left.dtype.kind not in "iu"
            or right.dtype.kind not in "iu" or left.size < 2
            or np.any(left < 0) or np.any(right < 0)
            or np.any(left >= len(x)) or np.any(right >= len(x))):
        raise ValueError("aligned valid pair indices and 2D representation required")
    if not np.isfinite(lipschitz_bound) or lipschitz_bound < 0:
        raise ValueError("finite nonnegative Lipschitz bound required")
    if (not np.isfinite(pinball_ratio_threshold) or pinball_ratio_threshold < 0
            or not np.isfinite(window_ratio_threshold) or window_ratio_threshold < 0):
        raise ValueError("finite nonnegative geometry thresholds required")
    scale = outcome_range_bound if selected_loss == "pinball" else window
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "geometry route requires a positive declared outcome span or window"}
    distance = np.mean(np.abs(x[left] - x[right]), axis=1)
    ratio = float(lipschitz_bound * np.median(distance) / scale)
    threshold = (
        pinball_ratio_threshold if selected_loss == "pinball"
        else window_ratio_threshold
    )
    grouped = ratio <= threshold
    route = decision_pair_group_envelope(
        x, outcomes, entities, left, right, groupings,
        loss=selected_loss, lipschitz_bound=lipschitz_bound,
        quantile=quantile, window=window,
        outcome_range_bound=outcome_range_bound, alpha=alpha,
        family_policy="grouped" if grouped else "paired",
        group_witness_rule="exact_lipschitz" if grouped else "radial",
        group_pinball_range_rule="finite_group" if grouped else "asymptotic",
        methods=(
            ("conditional_bentkus", "conditional_entity_mcd")
            if grouped and selected_loss == "window_miss"
            else ("conditional_bentkus",)
        ),
    )
    route = dict(route)
    route.update({
        "geometry_ratio": ratio,
        "geometry_ratio_threshold": threshold,
        "geometry_route_outcome_blind": True,
        "geometry_route_family": "grouped" if grouped else "paired",
        "geometry_thresholds_provenance_audited": False,
        "geometry_threshold_warning": (
            "the thresholds must be frozen before evaluation; Mayo retrospective "
            "outcomes influenced the window threshold and are not confirmatory"
        ),
    })
    return route
