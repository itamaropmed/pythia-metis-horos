"""Model-free attainable-risk bounds from outcome-blind near-replicate pairs."""
from __future__ import annotations

import numpy as np

from pythia.contracts import PairMatrix
from pythia.core.concentration import bentkus_lower_mean


def paired_replicate_lower_bound(
    y_left: np.ndarray,
    y_right: np.ndarray,
    distances: np.ndarray,
    left_entities: np.ndarray,
    right_entities: np.ndarray,
    pair_blocks: np.ndarray,
    *,
    loss: str,
    lipschitz_bound: float,
    distance_bound: float,
    outcome_difference_bound: float | None,
    outcome_term_clip: float | None = None,
    quantile: float | None = None,
    window: float | None = None,
    alpha: float = 0.05,
    concentration: str = "empirical_bernstein",
) -> dict:
    """Lower-bound conditional Bayes risk over disjoint paired covariates.

    Pair construction must use covariates only. Every sampling entity may occur
    at most once. Pairs are aggregated within randomized iid blocks and
    concentration is applied only across blocks, so matching dependence inside
    a block is allowed. ``conditional_hoeffding`` instead freezes the observed
    covariates and outcome-blind matching, then concentrates over the disjoint
    pairs; ``conditional_empirical_bernstein`` does the same with the
    independent/non-identical empirical Bernstein inequality. Both target the
    finite audited covariate population. For MAE the
    Lipschitz condition applies to a selected conditional median; for MSE it
    applies to the conditional mean.
    """
    yl = np.asarray(y_left, float).ravel()
    yr = np.asarray(y_right, float).ravel()
    d = np.asarray(distances, float).ravel()
    left = np.asarray(left_entities).astype(str).ravel()
    right = np.asarray(right_entities).astype(str).ravel()
    blocks = np.asarray(pair_blocks).astype(str).ravel()
    n = yl.size
    if n < 2 or any(values.size != n for values in (yr, d, left, right, blocks)):
        raise ValueError("paired replicate arrays must align and contain at least two pairs")
    if not np.isfinite(yl).all() or not np.isfinite(yr).all() or not np.isfinite(d).all():
        raise ValueError("paired replicate values must be finite")
    if np.any(d < 0):
        raise ValueError("pair distances must be nonnegative")
    entities = np.concatenate([left, right])
    if np.unique(entities).size != entities.size:
        return {
            "status": "refused", "claim_tier": "refused", "bound": None,
            "reason": "sampling entities overlap within or across pairs",
        }
    unique_blocks, block_inverse = np.unique(blocks, return_inverse=True)
    values = np.asarray([lipschitz_bound, distance_bound, alpha], float)
    if not np.isfinite(values).all() or lipschitz_bound < 0:
        raise ValueError("replicate guard values must be finite and Lipschitz nonnegative")
    if distance_bound <= 0:
        raise ValueError("replicate distance bound must be positive")
    loss_key = loss.lower()
    bounded_without_raw_cap = loss_key in {"window_miss", "w30_miss"}
    if outcome_difference_bound is None and outcome_term_clip is None and not bounded_without_raw_cap:
        raise ValueError("an outcome-difference cap or a bounded outcome-term clip is required")
    if outcome_difference_bound is not None and (
        not np.isfinite(outcome_difference_bound) or outcome_difference_bound <= 0
    ):
        raise ValueError("replicate outcome-difference bound must be finite and positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    pair_conditional_methods = {
        "conditional_hoeffding", "conditional_empirical_bernstein",
        "conditional_bentkus",
    }
    block_conditional_methods = {
        "conditional_block_hoeffding", "conditional_block_empirical_bernstein",
    }
    conditional_methods = pair_conditional_methods | block_conditional_methods
    if concentration not in {
        "empirical_bernstein", "hoeffding", *conditional_methods,
    }:
        raise ValueError(
            "concentration must be 'empirical_bernstein', 'hoeffding', "
            "or a pair/block conditional Hoeffding/empirical-Bernstein method"
        )
    if concentration not in conditional_methods and unique_blocks.size < 2:
        raise ValueError("block-level replicate concentration requires at least two independent blocks")
    if concentration in block_conditional_methods and unique_blocks.size < 2:
        raise ValueError("conditional block concentration requires at least two independent blocks")
    observed_distance = float(np.max(d))
    observed_outcome_difference = float(np.max(np.abs(yl - yr)))
    if observed_distance > distance_bound + 1e-12:
        return {
            "status": "refused", "claim_tier": "refused", "bound": None,
            "reason": "observed pair distance exceeds the declared cap",
            "observed_max_distance": observed_distance,
        }
    if (outcome_difference_bound is not None
            and observed_outcome_difference > outcome_difference_bound + 1e-12):
        return {
            "status": "refused", "claim_tier": "refused", "bound": None,
            "reason": "observed outcome difference exceeds the declared cap",
            "observed_max_outcome_difference": observed_outcome_difference,
        }

    if loss_key == "mae":
        outcome_terms = 0.5 * np.abs(yl - yr)
        penalties = 0.5 * lipschitz_bound * d
        scores = outcome_terms - penalties
        outcome_term_range = (
            0.5 * outcome_difference_bound
            if outcome_difference_bound is not None else float(outcome_term_clip)
        )
        score_min = -0.5 * lipschitz_bound * distance_bound
        score_max = outcome_term_range
        center_assumption = "a selected conditional median is Lipschitz in the declared metric"
    elif loss_key == "mse":
        outcome_terms = 0.5 * (yl - yr) ** 2
        penalties = 0.5 * (lipschitz_bound * d) ** 2
        scores = outcome_terms - penalties
        outcome_term_range = (
            0.5 * outcome_difference_bound ** 2
            if outcome_difference_bound is not None else float(outcome_term_clip)
        )
        score_min = -0.5 * (lipschitz_bound * distance_bound) ** 2
        score_max = outcome_term_range
        center_assumption = "the conditional mean is Lipschitz in the declared metric"
    elif loss_key in {"mae_pair_envelope", "pinball", "window_miss", "w30_miss", "mse_pair_envelope"}:
        shift = lipschitz_bound * d
        gap_after_shift = np.maximum(np.abs(yl - yr) - shift, 0.0)
        penalties = np.zeros_like(d)
        if loss_key == "mae_pair_envelope":
            outcome_terms = 0.5 * gap_after_shift
            term_cap = (
                0.5 * outcome_difference_bound
                if outcome_difference_bound is not None else float(outcome_term_clip)
            )
            center_assumption = "a selected conditional median is Lipschitz in the declared metric"
        elif loss_key == "pinball":
            if quantile is None or not np.isfinite(quantile) or not 0 < quantile < 1:
                raise ValueError("pinball loss requires a quantile strictly between 0 and 1")
            weight = min(quantile, 1.0 - quantile)
            outcome_terms = 0.5 * weight * gap_after_shift
            term_cap = (
                0.5 * weight * outcome_difference_bound
                if outcome_difference_bound is not None else float(outcome_term_clip)
            )
            center_assumption = "a selected conditional quantile is Lipschitz in the declared metric"
        elif loss_key == "mse_pair_envelope":
            outcome_terms = 0.25 * gap_after_shift ** 2
            term_cap = (
                0.25 * outcome_difference_bound ** 2
                if outcome_difference_bound is not None else float(outcome_term_clip)
            )
            center_assumption = "the conditional mean is Lipschitz in the declared metric"
        else:
            selected_window = 30.0 if loss_key == "w30_miss" and window is None else window
            if (selected_window is None or not np.isfinite(selected_window)
                    or selected_window <= 0):
                raise ValueError("window-miss loss requires a positive finite window")
            if loss_key == "w30_miss" and not np.isclose(selected_window, 30.0):
                raise ValueError("w30_miss fixes the window to 30 outcome units")
            outcome_terms = 0.5 * (
                np.abs(yl - yr) > 2.0 * selected_window + shift
            ).astype(float)
            term_cap = 0.5
            center_assumption = (
                "a selected conditional within-window coverage-maximizing center "
                "is Lipschitz in the declared metric"
            )
        scores = outcome_terms
        outcome_term_range = float(term_cap)
        score_min = 0.0
        score_max = float(term_cap)
    else:
        return {
            "status": "refused", "claim_tier": "refused", "bound": None,
            "reason": "paired-replicate theory does not support this loss",
        }
    if outcome_term_clip is not None:
        if (not np.isfinite(outcome_term_clip) or outcome_term_clip <= 0
                or outcome_term_clip > outcome_term_range):
            raise ValueError("outcome-term clip must be positive and no larger than its declared range")
        # E[min(A,c)] <= E[A], so clipping preserves the lower-risk inequality
        # while replacing the concentration range by c. This requires no tail
        # model or assertion that unseen outcomes stay below c.
        outcome_terms = np.minimum(outcome_terms, outcome_term_clip)
        scores = outcome_terms - penalties
        outcome_term_range = float(outcome_term_clip)
        score_max = float(outcome_term_clip)
    range_assumption = (
        "the window-miss pair indicator is bounded by 1/2 without a raw outcome-difference cap"
        if outcome_difference_bound is None and bounded_without_raw_cap else
        "the clipped outcome term is bounded by its predeclared clip; no raw outcome-difference cap is assumed"
        if outcome_difference_bound is None else
        "the declared outcome-difference cap holds almost surely"
    )
    block_counts = np.bincount(block_inverse).astype(float)
    equal_block_sizes = bool(np.all(block_counts == block_counts[0]))
    block_scores = np.bincount(
        block_inverse, weights=scores, minlength=unique_blocks.size
    ) / block_counts
    score_range = score_max - score_min
    log_term = np.log(2.0 / alpha)
    block_outcome_terms = np.bincount(
        block_inverse, weights=outcome_terms, minlength=unique_blocks.size
    ) / block_counts
    block_penalties = np.bincount(
        block_inverse, weights=penalties, minlength=unique_blocks.size
    ) / block_counts
    if concentration == "conditional_hoeffding":
        # Conditional on covariates and an outcome-blind fixed matching, each
        # score's penalty is deterministic and disjoint pairs have independent
        # outcome terms. One-sided Hoeffding therefore uses all pairs and only
        # the outcome-term range; iid covariates/blocks are not required.
        radius = outcome_term_range * np.sqrt(np.log(1.0 / alpha) / (2.0 * n))
    elif concentration == "conditional_empirical_bernstein":
        # Maurer--Pontil (2009), Theorem 11, applies to independent bounded
        # variables that need not be identically distributed. Apply its lower
        # tail to outcome_terms / outcome_term_range (equivalently to 1-X).
        radius = (
            np.sqrt(2.0 * np.var(outcome_terms, ddof=1) * log_term / n)
            + 7.0 * outcome_term_range * log_term / (3.0 * (n - 1))
        )
    elif concentration == "conditional_bentkus":
        radius = float(
            np.mean(outcome_terms)
            - bentkus_lower_mean(outcome_terms, outcome_term_range, alpha)
        )
    elif concentration == "conditional_block_hoeffding":
        radius = outcome_term_range * np.sqrt(
            np.log(1.0 / alpha) / (2.0 * unique_blocks.size)
        )
    elif concentration == "conditional_block_empirical_bernstein":
        radius = (
            np.sqrt(
                2.0 * np.var(block_outcome_terms, ddof=1) * log_term
                / unique_blocks.size
            )
            + 7.0 * outcome_term_range * log_term
            / (3.0 * (unique_blocks.size - 1))
        )
    elif concentration == "hoeffding":
        radius = score_range * np.sqrt(log_term / (2.0 * unique_blocks.size))
    else:
        radius = (
            np.sqrt(2.0 * np.var(block_scores, ddof=1) * log_term / unique_blocks.size)
            + 7.0 * score_range * log_term / (3.0 * (unique_blocks.size - 1))
        )
    raw_estimate = float(
        np.mean(block_outcome_terms - block_penalties)
        if concentration in block_conditional_methods else np.mean(block_scores)
    )
    # Risk is nonnegative. Preserve the raw corrected center for diagnostics,
    # but expose a feasible nonnegative plug-in so 0 <= bound <= estimate.
    estimate = float(max(raw_estimate, 0.0))
    bound = float(max(raw_estimate - radius, 0.0))
    if concentration in pair_conditional_methods:
        estimand = "average conditional Bayes risk over the audited pair endpoints"
        assumptions = (
            "the audited covariates and complete pairing are conditioned on as fixed",
            "sampling-entity outcomes are conditionally independent given their covariates",
            "pairing and metric use covariates only under a rule fixed before outcomes",
            "all pair endpoints are distinct sampling entities",
            center_assumption,
            range_assumption,
        )
        independent_units = n
    elif concentration in block_conditional_methods:
        estimand = (
            "average conditional Bayes risk over audited pair endpoints"
            if equal_block_sizes else
            "average conditional Bayes risk under equal weighting of audited blocks"
        )
        assumptions = (
            "the audited covariates, block assignment, and complete pairing are conditioned on as fixed",
            "outcome vectors are independent across declared blocks; dependence within blocks is unrestricted",
            "pairing and metric use covariates only under a rule fixed before outcomes",
            "all pair endpoints are distinct sampling entities",
            center_assumption,
            range_assumption,
            "unequal block sizes imply a block-weighted rather than entity-weighted estimand",
        )
        independent_units = unique_blocks.size
    else:
        estimand = "expected block-average conditional Bayes risk under the declared pairing design"
        assumptions = (
            "blocks are iid and generated or randomized independently of outcomes",
            "sampling-entity outcomes are conditionally independent given their covariates",
            "within-block pairing and metric use covariates only under a fixed rule",
            "all pair endpoints are distinct sampling entities",
            center_assumption,
            "the declared distance cap holds almost surely",
            range_assumption,
        )
        independent_units = unique_blocks.size
    return {
        "status": "complete", "claim_tier": "conditional", "loss": loss_key,
        "bound": bound, "estimate": estimate, "radius": float(radius),
        "raw_estimate": raw_estimate,
        "outcome_term_clip": outcome_term_clip,
        "outcome_difference_bound": outcome_difference_bound,
        "quantile": quantile if loss_key == "pinball" else None,
        "window": (30.0 if loss_key == "w30_miss" else window) if bounded_without_raw_cap else None,
        "estimate_role": "nonnegative corrected lower-bound center, not an identified Bayes-risk point estimate",
        "confidence": float(1.0 - alpha), "n_pairs": int(n),
        "n_independent_blocks": int(unique_blocks.size),
        "n_concentration_units": int(independent_units),
        "pairs_per_block_min": int(np.min(block_counts)),
        "pairs_per_block_max": int(np.max(block_counts)),
        "equal_block_sizes": equal_block_sizes,
        "mean_distance": float(np.mean(d)), "max_distance": observed_distance,
        "score_range": float(score_range),
        "concentration": concentration,
        "concentration_reference": (
            "Maurer and Pontil (2009), Theorem 11, arXiv:0907.3740"
            if concentration in {
                "conditional_empirical_bernstein",
                "conditional_block_empirical_bernstein",
            } else None
        ),
        "estimand": estimand,
        "assumptions": assumptions,
    }


def paired_replicate_concentration_envelope(
    y_left: np.ndarray,
    y_right: np.ndarray,
    distances: np.ndarray,
    left_entities: np.ndarray,
    right_entities: np.ndarray,
    pair_blocks: np.ndarray,
    *,
    loss: str,
    lipschitz_bound: float,
    distance_bound: float,
    outcome_difference_bound: float | None,
    quantile: float | None = None,
    window: float | None = None,
    alpha: float = 0.05,
    methods: tuple[str, ...] = (
        "conditional_hoeffding", "conditional_empirical_bernstein",
    ),
) -> dict:
    """Select the sharpest valid bound from a predeclared method registry."""
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rows = [paired_replicate_lower_bound(
        y_left, y_right, distances, left_entities, right_entities, pair_blocks,
        loss=loss, lipschitz_bound=lipschitz_bound,
        distance_bound=distance_bound,
        outcome_difference_bound=outcome_difference_bound,
        quantile=quantile, window=window, alpha=alpha / len(methods),
        concentration=method,
    ) for method in methods]
    if any(row["status"] != "complete" for row in rows):
        return next(row for row in rows if row["status"] != "complete")
    selected = int(np.argmax([row["bound"] for row in rows]))
    best = dict(rows[selected])
    best.update({
        "confidence": float(1.0 - alpha), "simultaneous": True,
        "selected_method": methods[selected],
        "predeclared_methods": list(methods),
        "alpha_allocation": "Bonferroni over predeclared concentration methods",
        "method_rows": [
            {"method": method, "bound": row["bound"],
             "estimate": row["estimate"], "radius": row["radius"]}
            for method, row in zip(methods, rows)
        ],
    })
    return best


def paired_replicate_window_curve(
    y_left: np.ndarray,
    y_right: np.ndarray,
    distances: np.ndarray,
    left_entities: np.ndarray,
    right_entities: np.ndarray,
    pair_blocks: np.ndarray,
    *,
    windows: np.ndarray,
    lipschitz_bound: float,
    distance_bound: float,
    alpha: float = 0.05,
    concentration: str = "conditional_hoeffding",
    absolute_error_quantiles: tuple[float, ...] = (),
) -> dict:
    """Simultaneous miss-risk bounds and implied absolute-error quantile floors.

    An error quantile lower floor is conservative: if every predictor has
    P(|Y-a(X)| > w) > 1-p, then its p-quantile of absolute error exceeds w.
    The Lipschitz center assumption must hold for every window in the grid.
    """
    grid = np.asarray(windows, float).ravel()
    if (grid.size == 0 or not np.isfinite(grid).all() or np.any(grid <= 0)
            or np.any(np.diff(grid) <= 0)):
        raise ValueError("windows must be nonempty, positive, and strictly increasing")
    levels = np.asarray(absolute_error_quantiles, float).ravel()
    if np.any(~np.isfinite(levels)) or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("absolute-error quantiles must be in (0, 1)")
    rows = [paired_replicate_lower_bound(
        y_left, y_right, distances, left_entities, right_entities, pair_blocks,
        loss="window_miss", window=float(value),
        lipschitz_bound=lipschitz_bound, distance_bound=distance_bound,
        outcome_difference_bound=None, alpha=alpha / grid.size,
        concentration=concentration,
    ) for value in grid]
    if any(row["status"] != "complete" for row in rows):
        return next(row for row in rows if row["status"] != "complete")
    return {
        "status": "complete", "claim_tier": "conditional",
        "confidence": float(1.0 - alpha), "simultaneous": True,
        "alpha_allocation": "Bonferroni over predeclared window grid",
        "curve": [
            {"window": float(w), "minimum_miss_risk": row["bound"],
             "maximum_attainable_within_window_percent": float(100 * (1 - row["bound"]))}
            for w, row in zip(grid, rows)
        ],
        "absolute_error_quantile_lower_floors": [
            {"quantile": float(level), "strict_lower_floor": float(max(
                [w for w, row in zip(grid, rows) if row["bound"] > 1 - level],
                default=0.0,
            ))}
            for level in levels
        ],
        "assumptions": rows[0]["assumptions"] + (
            "the selected coverage-maximizing centers share the declared Lipschitz cap at every grid window",
        ),
    }


def paired_replicate_clipped_envelope(
    y_left: np.ndarray,
    y_right: np.ndarray,
    distances: np.ndarray,
    left_entities: np.ndarray,
    right_entities: np.ndarray,
    pair_blocks: np.ndarray,
    *,
    loss: str,
    lipschitz_bound: float,
    distance_bound: float,
    outcome_difference_bound: float | None,
    outcome_term_clips: tuple[float, ...],
    quantile: float | None = None,
    window: float | None = None,
    alpha: float = 0.05,
    concentration: str = "conditional_empirical_bernstein",
) -> dict:
    """Selection-safe maximum over predeclared clipped-term lower bounds.

    The caller must freeze the clip grid without using the evaluation
    outcomes. Bonferroni protects selection among clips. For ordinary MAE/MSE
    conditional methods, each clip's outcome-term event covers every L value;
    pair-envelope losses have L-dependent terms and need a separate L-grid
    multiplicity correction.
    """
    clips = np.asarray(outcome_term_clips, float).ravel()
    if (clips.size == 0 or not np.isfinite(clips).all()
            or np.any(clips <= 0) or np.any(np.diff(clips) <= 0)):
        raise ValueError("outcome-term clips must be nonempty, finite, positive, and strictly increasing")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rows = [
        paired_replicate_lower_bound(
            y_left, y_right, distances, left_entities, right_entities,
            pair_blocks, loss=loss, lipschitz_bound=lipschitz_bound,
            distance_bound=distance_bound,
            outcome_difference_bound=outcome_difference_bound,
            outcome_term_clip=float(clip), alpha=alpha / clips.size,
            concentration=concentration, quantile=quantile, window=window,
        )
        for clip in clips
    ]
    if any(row["status"] != "complete" for row in rows):
        refusal = next(row for row in rows if row["status"] != "complete")
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": refusal["reason"]}
    selected = int(np.argmax([row["bound"] for row in rows]))
    best = dict(rows[selected])
    best.update({
        "confidence": float(1.0 - alpha),
        "selected_clip": float(clips[selected]),
        "predeclared_clips": clips.tolist(),
        "clip_selection_simultaneous": True,
        "clip_alpha_allocation": "Bonferroni over predeclared clip grid",
        "clip_rows": [
            {"clip": float(clip), "bound": row["bound"],
             "estimate": row["estimate"], "radius": row["radius"]}
            for clip, row in zip(clips, rows)
        ],
    })
    return best


def paired_replicate_sensitivity_curve(
    y_left: np.ndarray,
    y_right: np.ndarray,
    distances: np.ndarray,
    left_entities: np.ndarray,
    right_entities: np.ndarray,
    pair_blocks: np.ndarray,
    *,
    loss: str,
    lipschitz_bounds: np.ndarray,
    distance_bound: float,
    outcome_difference_bound: float | None,
    outcome_term_clip: float | None = None,
    quantile: float | None = None,
    window: float | None = None,
    alpha: float = 0.05,
    concentration: str = "empirical_bernstein",
) -> dict:
    """Return a Bonferroni-simultaneous smoothness sensitivity curve."""
    grid = np.asarray(lipschitz_bounds, float).ravel()
    if grid.size == 0 or not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError("lipschitz_bounds must be a nonempty nonnegative finite grid")
    if np.any(np.diff(grid) < 0):
        raise ValueError("lipschitz_bounds must be sorted")
    shared_conditional_event = (
        concentration.startswith("conditional_") and loss.lower() in {"mae", "mse"}
    )
    event_alpha = alpha if shared_conditional_event else alpha / grid.size
    rows = [
        paired_replicate_lower_bound(
            y_left, y_right, distances, left_entities, right_entities,
            pair_blocks, loss=loss, lipschitz_bound=float(value),
            distance_bound=distance_bound,
            outcome_difference_bound=outcome_difference_bound,
            outcome_term_clip=outcome_term_clip,
            quantile=quantile, window=window,
            alpha=event_alpha, concentration=concentration,
        )
        for value in grid
    ]
    if any(row["status"] != "complete" for row in rows):
        refusal = next(row for row in rows if row["status"] != "complete")
        return {
            "status": "refused", "claim_tier": "refused",
            "reason": refusal["reason"],
        }
    return {
        "status": "complete", "claim_tier": "conditional",
        "confidence": float(1.0 - alpha), "simultaneous": True,
        "alpha_allocation": (
            "one shared outcome-term event covers the complete Lipschitz grid"
            if shared_conditional_event else "Bonferroni over Lipschitz grid"
        ),
        "curve": [
            {
                "lipschitz_bound": float(value), "bound": row["bound"],
                "estimate": row["estimate"], "radius": row["radius"],
            }
            for value, row in zip(grid, rows)
        ],
        "assumptions": rows[0]["assumptions"],
        "n_pairs": rows[0]["n_pairs"],
        "n_independent_blocks": rows[0]["n_independent_blocks"],
    }


def paired_replicate_ensemble_sensitivity(
    designs: tuple[PairMatrix, ...],
    *,
    loss: str,
    lipschitz_bounds: np.ndarray,
    distance_bound: float,
    outcome_difference_bound: float | None,
    outcome_term_clip: float | None = None,
    quantile: float | None = None,
    window: float | None = None,
    alpha: float = 0.05,
    concentration: str = "empirical_bernstein",
) -> dict:
    """Simultaneously select over a predeclared registry of pair designs."""
    if not designs:
        raise ValueError("at least one pair design is required")
    grid = np.asarray(lipschitz_bounds, float).ravel()
    if grid.size == 0 or not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError("lipschitz_bounds must be a nonempty nonnegative finite grid")
    if np.any(np.diff(grid) < 0):
        raise ValueError("lipschitz_bounds must be sorted")

    reference = None
    for design in designs:
        design.validate()
        entity = np.concatenate([
            np.asarray(design.left_entities).astype(str),
            np.asarray(design.right_entities).astype(str),
        ])
        outcome = np.concatenate([
            np.asarray(design.y_left, float), np.asarray(design.y_right, float),
        ])
        order = np.argsort(entity)
        aligned = (entity[order], outcome[order])
        if np.unique(aligned[0]).size != aligned[0].size:
            return {
                "status": "refused", "claim_tier": "refused",
                "reason": "every ensemble design must use each entity exactly once",
            }
        if reference is None:
            reference = aligned
        elif (not np.array_equal(aligned[0], reference[0]) or
              not np.allclose(aligned[1], reference[1], rtol=0, atol=0)):
            return {
                "status": "refused", "claim_tier": "refused",
                "reason": "ensemble pair designs do not cover the identical entity/outcome population",
            }

    conditional = (
        concentration.startswith("conditional_") and loss.lower() in {"mae", "mse"}
    )
    event_alpha = (
        alpha / len(designs) if conditional
        else alpha / (len(designs) * grid.size)
    )
    evaluations = []
    for design_index, design in enumerate(designs):
        for value in grid:
            result = paired_replicate_lower_bound(
                design.y_left, design.y_right, design.distances,
                design.left_entities, design.right_entities, design.pair_blocks,
                loss=loss, lipschitz_bound=float(value),
                distance_bound=distance_bound,
                outcome_difference_bound=outcome_difference_bound,
                outcome_term_clip=outcome_term_clip,
                quantile=quantile, window=window,
                alpha=event_alpha, concentration=concentration,
            )
            if result["status"] != "complete":
                return result
            evaluations.append((design_index, float(value), result))

    curve = []
    for value in grid:
        candidates = [row for row in evaluations if row[1] == float(value)]
        selected = max(candidates, key=lambda row: row[2]["bound"])
        curve.append({
            "lipschitz_bound": float(value), "bound": selected[2]["bound"],
            "estimate": selected[2]["estimate"], "radius": selected[2]["radius"],
            "selected_design_index": int(selected[0]),
            "selected_design_hash": designs[selected[0]].meta.get("design_hash"),
        })
    return {
        "status": "complete", "claim_tier": "conditional",
        "confidence": float(1.0 - alpha), "simultaneous": True,
        "selection_safe": True, "n_designs": len(designs),
        "n_lipschitz_values": int(grid.size),
        "alpha_allocation": (
            "Bonferroni over designs; one shared outcome-term event covers the Lipschitz grid"
            if conditional else
            "Bonferroni over predeclared designs and Lipschitz grid"
        ),
        "curve": curve,
        "assumptions": evaluations[0][2]["assumptions"] + (
            "all pair designs cover the identical entity population exactly once",
            "the complete seed/design registry was fixed without outcomes",
        ),
    }
