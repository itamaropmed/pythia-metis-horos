"""Conditional multi-outcome lower-risk witnesses with no predictor fit."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from pythia.core.concentration import bentkus_lower_mean


def _exact_lipschitz_pinball_score(
    features: np.ndarray, outcomes: np.ndarray, lipschitz_bound: float,
    quantile: float,
) -> float:
    """Minimum empirical pinball loss over all groupwise Lipschitz actions.

    The action variables are not a fitted predictor: they exist only inside
    this outcome score. The true conditional quantile action vector is a
    feasible comparator under the declared Lipschitz assumption.
    """
    size = len(outcomes)
    objective = np.r_[np.zeros(size), np.full(size, 1 / size)]
    constraints = []
    limits = []
    for i in range(size):
        upper_residual = np.zeros(2 * size)
        upper_residual[i] = -quantile
        upper_residual[size + i] = -1
        constraints.append(upper_residual)
        limits.append(-quantile * outcomes[i])
        lower_residual = np.zeros(2 * size)
        lower_residual[i] = 1 - quantile
        lower_residual[size + i] = -1
        constraints.append(lower_residual)
        limits.append((1 - quantile) * outcomes[i])
    for i in range(size):
        for j in range(i + 1, size):
            cap = lipschitz_bound * np.mean(np.abs(features[i] - features[j]))
            edge = np.zeros(2 * size)
            edge[i], edge[j] = 1, -1
            constraints.extend((edge, -edge))
            limits.extend((cap, cap))
    low, high = float(np.min(outcomes)), float(np.max(outcomes))
    result = linprog(
        objective, A_ub=np.asarray(constraints), b_ub=np.asarray(limits),
        bounds=[(low, high)] * size + [(0, None)] * size, method="highs",
    )
    if not result.success or not np.isfinite(result.fun):
        raise ArithmeticError(f"exact group witness solver failed: {result.message}")
    return float(max(0., result.fun))


def _exact_lipschitz_window_score(
    features: np.ndarray, outcomes: np.ndarray, lipschitz_bound: float,
    window: float,
) -> float:
    """Exact minimum window-miss fraction under group Lipschitz actions.

    A set of cases can all be in-window iff each pair obeys
    ``|y_i-y_j| <= 2*window + L*d(x_i,x_j)``. For scalar action intervals
    and a metric, these pairwise conditions are sufficient for a Lipschitz
    selection; maximum simultaneous coverage is therefore a maximum clique.
    """
    size = len(outcomes)
    if size > 24:
        raise ArithmeticError("exact window witness supports groups of at most 24")
    adjacency = [0] * size
    for i in range(size):
        for j in range(i + 1, size):
            cap = 2 * window + lipschitz_bound * np.mean(
                np.abs(features[i] - features[j])
            )
            if abs(outcomes[i] - outcomes[j]) <= cap + 1e-12:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    best = 0

    def search(candidates: int, chosen: int) -> None:
        nonlocal best
        if chosen + candidates.bit_count() <= best:
            return
        while candidates:
            bit = candidates & -candidates
            vertex = bit.bit_length() - 1
            candidates ^= bit
            search(candidates & adjacency[vertex], chosen + 1)
            if chosen + candidates.bit_count() <= best:
                return
        best = max(best, chosen)

    search((1 << size) - 1, 0)
    return float(1 - best / size)


def grouped_decision_lower_bound(
    representation: np.ndarray,
    outcomes: np.ndarray,
    entities: np.ndarray,
    groups: np.ndarray,
    *,
    loss: str,
    lipschitz_bound: float,
    outcome_range_bound: float | None = None,
    outcome_term_clip: float | None = None,
    quantile: float | None = None,
    window: float | None = None,
    alpha: float = .05,
    concentration: str = "conditional_hoeffding",
    center_rule: str = "mean",
    witness_rule: str = "radial",
    pinball_range_rule: str = "asymptotic",
) -> dict:
    """Bound finite-design Bayes risk with radial or exact group actions.

    For group centroid c, pairwise Lipschitz Bayes actions admit a common
    *virtual* scalar action within L d(x,c) of every observed action: the
    intervals [a_i-Ld_i, a_i+Ld_i] intersect by the metric triangle
    inequality. The centroid need not represent a feasible patient. Relaxing
    each observed action to that interval gives the radial lower witness.
    The exact option instead minimizes observed loss over all groupwise
    Lipschitz action vectors. Groups are disjoint and must have been formed
    without outcomes. Outcomes must be conditionally independent given the
    audited covariates.
    """
    x = np.asarray(representation, float)
    y = np.asarray(outcomes, float).ravel()
    entity = np.asarray(entities).astype(str).ravel()
    raw_group = np.asarray(groups)
    if not np.issubdtype(raw_group.dtype, np.integer):
        raise ValueError("group indices must have an integer dtype")
    group = raw_group.astype(int, copy=False)
    if (x.ndim != 2 or group.ndim != 2 or x.shape[0] != y.size
            or y.size != entity.size or group.shape[1] < 2
            or group.shape[0] < 2 or group.size != y.size):
        raise ValueError("groups must partition all aligned rows into equal groups")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and outcomes must be finite")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("representation must be in [0,1]")
    if (np.unique(entity).size != y.size or np.any(np.sort(group.ravel()) != np.arange(y.size))):
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "groups overlap, omit, or repeat sampling entities"}
    if not np.isfinite(lipschitz_bound) or lipschitz_bound < 0 or not 0 < alpha < 1:
        raise ValueError("valid Lipschitz cap and alpha required")
    if concentration not in {
        "conditional_hoeffding", "conditional_empirical_bernstein", "conditional_bentkus",
        "conditional_entity_mcd",
    }:
        raise ValueError("unsupported grouped concentration")
    if center_rule not in {"mean", "coordinate_median"}:
        raise ValueError("center_rule must be mean or coordinate_median")
    if witness_rule not in {"radial", "exact_lipschitz"}:
        raise ValueError("witness_rule must be radial or exact_lipschitz")
    if pinball_range_rule not in {"asymptotic", "finite_group"}:
        raise ValueError("pinball_range_rule must be asymptotic or finite_group")
    delta = None
    if witness_rule == "radial":
        center = (
            np.mean(x[group], axis=1) if center_rule == "mean"
            else np.median(x[group], axis=1)
        )
        delta = lipschitz_bound * np.mean(
            np.abs(x[group] - center[:, None, :]), axis=2
        )
    selected = loss.lower()
    scores = np.empty(len(group), float)
    if selected == "window_miss":
        if window is None or not np.isfinite(window) or window <= 0:
            raise ValueError("positive window required")
        for i, rows in enumerate(group):
            if witness_rule == "exact_lipschitz":
                try:
                    scores[i] = _exact_lipschitz_window_score(
                        x[rows], y[rows], lipschitz_bound, window,
                    )
                except ArithmeticError as exc:
                    return {"status": "refused", "claim_tier": "refused", "bound": None,
                            "reason": str(exc)}
                continue
            low = y[rows] - window - delta[i]
            high = y[rows] + window + delta[i]
            # At a shared latent action, a case can avoid a miss only inside
            # its expanded outcome interval. Max overlap is exact by sweep.
            endpoints = np.concatenate([low, high])
            changes = np.concatenate([np.ones(len(rows)), -np.ones(len(rows))])
            order = np.lexsort((-changes, endpoints))  # starts before ends at ties
            overlap = int(np.max(np.cumsum(changes[order])))
            scores[i] = 1.0 - overlap / len(rows)
        score_range = 1.0 - 1.0 / group.shape[1]
        center_assumption = "a coverage-maximizing center is Lipschitz in the group metric"
    elif selected in {"pinball", "mae"}:
        if (outcome_range_bound is None and outcome_term_clip is None):
            raise ValueError("MAE/pinball group scores need a range cap or predeclared clip")
        if (outcome_range_bound is not None and (
                not np.isfinite(outcome_range_bound) or outcome_range_bound <= 0)):
            raise ValueError("outcome range cap must be finite and positive")
        if (outcome_range_bound is not None
                and np.max(y) - np.min(y) > outcome_range_bound + 1e-12):
            return {"status": "refused", "claim_tier": "refused", "bound": None,
                    "reason": "observed outcome range exceeds declared cap"}
        if selected == "pinball":
            if quantile is None or not np.isfinite(quantile) or not 0 < quantile < 1:
                raise ValueError("pinball quantile must be in (0,1)")
        for i, rows in enumerate(group):
            if witness_rule == "exact_lipschitz":
                level = quantile if selected == "pinball" else .5
                try:
                    exact = _exact_lipschitz_pinball_score(
                        x[rows], y[rows], lipschitz_bound, level,
                    )
                except ArithmeticError as exc:
                    return {"status": "refused", "claim_tier": "refused", "bound": None,
                            "reason": str(exc)}
                scores[i] = exact if selected == "pinball" else 2 * exact
                continue
            candidates = np.concatenate([y[rows] - delta[i], y[rows] + delta[i]])
            residual = y[rows][None, :] - candidates[:, None]
            if selected == "pinball":
                cost = (
                    quantile * np.maximum(residual - delta[i][None, :], 0)
                    + (1 - quantile) * np.maximum(-residual - delta[i][None, :], 0)
                )
            else:
                cost = np.maximum(np.abs(residual) - delta[i][None, :], 0)
            scores[i] = float(np.min(np.mean(cost, axis=1)))
        if selected == "pinball":
            size = group.shape[1]
            low_counts = np.arange(size + 1)
            finite_group_factor = float(np.max(np.minimum(
                quantile * (size - low_counts) / size,
                (1 - quantile) * low_counts / size,
            )))
            score_range = (
                (finite_group_factor if pinball_range_rule == "finite_group"
                 else quantile * (1 - quantile)) * outcome_range_bound
                if outcome_range_bound is not None else float("inf")
            )
            center_assumption = "a selected conditional quantile is Lipschitz in the group metric"
        else:
            score_range = (
                .5 * outcome_range_bound
                if outcome_range_bound is not None else float("inf")
            )
            center_assumption = "a selected conditional median is Lipschitz in the group metric"
    else:
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "group witness does not support this loss"}
    if outcome_term_clip is not None:
        if (selected == "window_miss" or not np.isfinite(outcome_term_clip)
                or outcome_term_clip <= 0 or outcome_term_clip > score_range):
            raise ValueError("clip must be positive, finite, within the score range, and for MAE/pinball")
        scores = np.minimum(scores, outcome_term_clip)
        score_range = float(outcome_term_clip)
    n = len(scores)
    if concentration == "conditional_hoeffding":
        radius = score_range * np.sqrt(np.log(1 / alpha) / (2 * n))
    elif concentration == "conditional_empirical_bernstein":
        log_term = np.log(2 / alpha)
        radius = (
            np.sqrt(2 * np.var(scores, ddof=1) * log_term / n)
            + 7 * score_range * log_term / (3 * (n - 1))
        )
    elif concentration == "conditional_bentkus":
        radius = float(np.mean(scores) - bentkus_lower_mean(scores, score_range, alpha))
    else:
        if selected == "window_miss":
            entity_sensitivity = 1.
        elif outcome_range_bound is not None:
            entity_sensitivity = outcome_range_bound * (
                max(quantile, 1 - quantile) if selected == "pinball" else 1.
            )
        else:
            return {"status": "refused", "claim_tier": "refused", "bound": None,
                    "reason": "entity bounded differences needs a declared raw outcome range"}
        # Replacing one entity's outcome changes the group minimum by at
        # most entity_sensitivity / group_size and the cohort mean by at
        # most entity_sensitivity / number_of_entities.
        radius = entity_sensitivity * np.sqrt(np.log(1 / alpha) / (2 * y.size))
    center_value = float(np.mean(scores))
    lp_unverified = witness_rule == "exact_lipschitz" and selected in {"mae", "pinball"}
    return {
        "status": "complete",
        "claim_tier": "exploratory" if lp_unverified else "conditional",
        "loss": selected,
        "bound": float(max(0.0, center_value - radius)),
        "estimate": center_value, "radius": float(radius),
        "score_range": float(score_range),
        "n_groups": int(n), "group_size": int(group.shape[1]),
        "mean_group_action_radius": (
            float(np.mean(delta)) if witness_rule == "radial" else None
        ),
        "outcome_blind_center_rule": (
            center_rule if witness_rule == "radial" else None
        ),
        "witness_rule": witness_rule,
        "numerical_solver_audit_status": (
            "not_formally_verified" if lp_unverified else
            "exact_clique_search" if witness_rule == "exact_lipschitz" else
            "not_applicable"
        ),
        "outcome_term_clip": outcome_term_clip,
        "outcome_range_bound": outcome_range_bound,
        "finite_group_pinball_range_factor": (
            finite_group_factor if selected == "pinball" else None
        ),
        "pinball_range_rule": pinball_range_rule,
        "concentration": concentration,
        "estimand": "average conditional Bayes risk over all audited group endpoints",
        "assumptions": (
            "groups were fixed from covariates without outcomes",
            "outcomes are conditionally independent across distinct sampling entities",
            center_assumption,
            "window-miss witness is bounded without an outcome range cap"
            if selected == "window_miss" else
            "the clipped witness is bounded without a raw outcome range cap"
            if outcome_range_bound is None else
            "the declared outcome range holds almost surely",
        ),
    }


def grouped_decision_envelope(
    representation: np.ndarray,
    outcomes: np.ndarray,
    entities: np.ndarray,
    groupings: dict[int, np.ndarray],
    *,
    loss: str,
    lipschitz_bound: float,
    outcome_range_bound: float | None = None,
    outcome_term_clips: tuple[float, ...] | None = None,
    quantile: float | None = None,
    window: float | None = None,
    alpha: float = .05,
    methods: tuple[str, ...] = (
        "conditional_hoeffding", "conditional_empirical_bernstein",
    ),
    center_rule: str = "mean",
    witness_rule: str = "radial",
    pinball_range_rule: str = "asymptotic",
) -> dict:
    """Bonferroni-safe choice over predeclared group sizes and methods."""
    if not groupings or not methods or len(set(methods)) != len(methods):
        raise ValueError("nonempty groupings and unique methods required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    clips = (None,) if outcome_term_clips is None else tuple(outcome_term_clips)
    if not clips or (outcome_term_clips is not None and (
            any(not np.isfinite(value) or value <= 0 for value in clips)
            or any(right <= left for left, right in zip(clips, clips[1:])))):
        raise ValueError("clip registry must be positive and strictly increasing")
    if any(size != np.asarray(grouping).shape[1] for size, grouping in groupings.items()):
        raise ValueError("grouping keys must equal their group sizes")
    rows = [grouped_decision_lower_bound(
        representation, outcomes, entities, grouping,
        loss=loss, lipschitz_bound=lipschitz_bound,
        outcome_range_bound=outcome_range_bound,
        outcome_term_clip=clip,
        quantile=quantile, window=window,
        center_rule=center_rule,
        witness_rule=witness_rule,
        pinball_range_rule=pinball_range_rule,
        alpha=alpha / (len(groupings) * len(methods) * len(clips)),
        concentration=method,
    )
        for grouping in groupings.values() for method in methods for clip in clips
    ]
    if any(row["status"] != "complete" for row in rows):
        return next(row for row in rows if row["status"] != "complete")
    selected = max(rows, key=lambda row: row["bound"])
    result = dict(selected)
    result.update({
        "confidence": float(1 - alpha), "simultaneous": True,
        "predeclared_group_sizes": sorted(groupings),
        "predeclared_methods": list(methods),
        "predeclared_clips": None if outcome_term_clips is None else list(clips),
        "alpha_allocation": "Bonferroni over group-size, concentration, and clip registry",
        "selection_adjusted_within_supplied_registry": True,
        "registry_outcome_blind_provenance_audited": False,
        "registry_warning": (
            "multiplicity correction is valid only if the complete grouping and "
            "method registry was fixed without using evaluation outcomes"
        ),
    })
    return result
