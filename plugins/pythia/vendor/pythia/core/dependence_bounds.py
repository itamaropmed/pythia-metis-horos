"""Transparent squared-loss bounds under capped shared predictor error.

These functions do not train models. They consume risks and disagreements, or
frozen predictions. The population result is algebraic. The finite-sample
result adds simultaneous one-sided Hoeffding bounds and therefore requires a
genuinely independent evaluation sample plus a declared almost-sure error cap.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import brentq


def sharp_pairwise_population_bound(
    risks: np.ndarray,
    disagreements: np.ndarray,
    lambda_max: float,
    *,
    tolerance: float = 1e-10,
) -> dict:
    """Lower-bound squared-loss Bayes risk from exact population moments.

    Write predictor errors relative to the Bayes act as ``e_i`` and assume
    ``Corr(e_i,e_j) <= lambda_max`` for every pair. For pair ``i,j``, observed
    risks identify ``a_i-a_j`` while disagreement identifies
    ``a_i+a_j-2 Cov(e_i,e_j)``. Solving the correlation inequality gives the
    largest compatible pairwise excess-risk sum and hence a lower risk bound.
    The maximum of the valid pair bounds is returned.
    """
    r = np.asarray(risks, float).ravel()
    d = np.asarray(disagreements, float)
    if r.size < 2 or d.shape != (r.size, r.size):
        raise ValueError("need a square disagreement matrix for at least two predictors")
    if not np.isfinite(r).all() or not np.isfinite(d).all() or np.any(r < 0) or np.any(d < -tolerance):
        raise ValueError("risks and disagreements must be finite and nonnegative")
    if not 0 <= lambda_max < 1:
        raise ValueError("lambda_max must be in [0, 1)")
    if not np.allclose(d, d.T, atol=tolerance) or not np.allclose(np.diag(d), 0, atol=tolerance):
        raise ValueError("disagreement matrix must be symmetric with a zero diagonal")

    rows = []
    for i, j in combinations(range(r.size), 2):
        dij = max(float(d[i, j]), 0.0)
        delta = float(r[i] - r[j])
        radicand = dij * dij - (1.0 - lambda_max * lambda_max) * delta * delta
        if radicand < -tolerance:
            rows.append({"pair": (i, j), "feasible": False, "radicand": radicand})
            continue
        root = np.sqrt(max(radicand, 0.0))
        if lambda_max == 0:
            max_excess_sum = dij
        else:
            max_excess_sum = (
                dij + lambda_max * root
            ) / (1.0 - lambda_max * lambda_max)
        feasible = max_excess_sum + tolerance >= abs(delta)
        bound = 0.5 * (float(r[i] + r[j]) - max_excess_sum)
        rows.append({
            "pair": (i, j), "feasible": bool(feasible),
            "risk_difference": delta, "disagreement": dij,
            "max_excess_sum": float(max_excess_sum),
            "bound": float(max(bound, 0.0)), "radicand": float(radicand),
        })
    feasible_rows = [row for row in rows if row.get("feasible")]
    if len(feasible_rows) != len(rows):
        return {
            "status": "refused", "claim_tier": "refused", "bound": None,
            "lambda_max": float(lambda_max), "pairs": rows,
            "reason": "at least one pair of population moments is infeasible under the supplied correlation cap",
        }
    active = max(feasible_rows, key=lambda row: row["bound"])
    return {
        "status": "complete", "claim_tier": "conditional",
        "bound": float(active["bound"]), "lambda_max": float(lambda_max),
        "active_pair": active["pair"], "pairs": rows,
        "assumptions": (
            "squared loss with exact population risks and disagreements",
            "each predictor error has finite second moment",
            "every pairwise predictor-error correlation is at most lambda_max",
        ),
    }


def _sharp_rectangle_lower_bound(
    risk_lower_i: float,
    risk_upper_i: float,
    risk_lower_j: float,
    risk_upper_j: float,
    disagreement_upper: float,
    lambda_max: float,
) -> float | None:
    """Minimise the sharp population formula over a moment rectangle.

    This is analytic: after parameterising by the risk difference, the
    objective is convex on each side of the minimum feasible risk sum. Its only
    possible interior stationary points are ``delta=+/- disagreement``.
    ``None`` means the rectangle has no moments feasible under the supplied cap.
    """
    c = 1.0 - lambda_max * lambda_max
    d = max(float(disagreement_upper), 0.0)
    delta_lo = max(risk_lower_i - risk_upper_j, -d / np.sqrt(c))
    delta_hi = min(risk_upper_i - risk_lower_j, d / np.sqrt(c))
    if delta_lo > delta_hi:
        return None
    kink = risk_lower_i - risk_lower_j
    candidates = [delta_lo, delta_hi, np.clip(kink, delta_lo, delta_hi)]
    for stationary in (-d, d):
        if delta_lo <= stationary <= delta_hi:
            candidates.append(stationary)

    values = []
    for delta in candidates:
        risk_sum = max(2.0 * risk_lower_i - delta, 2.0 * risk_lower_j + delta)
        if risk_sum > min(2.0 * risk_upper_i - delta, 2.0 * risk_upper_j + delta) + 1e-10:
            continue
        radicand = max(d * d - c * delta * delta, 0.0)
        max_excess_sum = d if lambda_max == 0 else (
            d + lambda_max * np.sqrt(radicand)
        ) / c
        values.append(0.5 * (risk_sum - max_excess_sum))
    return None if not values else float(max(min(values), 0.0))


def bounded_sample_pairwise_bound(
    y: np.ndarray,
    predictions: np.ndarray,
    lambda_max: float,
    absolute_error_bound: float,
    *,
    alpha: float = 0.05,
    concentration: str = "empirical_bernstein",
) -> dict:
    """Finite-sample lower confidence bound on squared-loss Bayes risk.

    This conservative rung uses ``a_i+a_j <= d_ij/(1-lambda_max)`` and
    simultaneous one-sided Hoeffding bounds. It is valid only for fixed
    predictors evaluated on independent rows and when
    ``|Y-f_i(X)| <= absolute_error_bound`` almost surely for every predictor.
    Ordinary OOF rows generally are not independent enough for this theorem.
    """
    y = np.asarray(y, float).ravel()
    p = np.atleast_2d(np.asarray(predictions, float))
    if p.shape[1] != y.size or p.shape[0] < 2 or y.size < 2:
        raise ValueError("need at least two aligned predictors and two evaluation rows")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("evaluation arrays must be finite")
    if not 0 <= lambda_max < 1:
        raise ValueError("lambda_max must be in [0, 1)")
    if not np.isfinite(absolute_error_bound) or absolute_error_bound <= 0:
        raise ValueError("absolute_error_bound must be finite and positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    observed_max = float(np.max(np.abs(p - y[None, :])))
    if observed_max > absolute_error_bound + 1e-12:
        return {
            "status": "refused", "claim_tier": "refused", "bound": None,
            "reason": "observed prediction error exceeds the declared almost-sure cap",
            "observed_max_error": observed_max,
        }

    b = p.shape[0]
    pair_indices = tuple(combinations(range(b), 2))
    n_events = b + len(pair_indices)
    empirical_risk_samples = (p - y[None, :]) ** 2
    empirical_risks = np.mean(empirical_risk_samples, axis=1)

    def radius(values: np.ndarray, value_range: float) -> float:
        if concentration == "hoeffding":
            return float(value_range * np.sqrt(np.log(n_events / alpha) / (2.0 * y.size)))
        if concentration != "empirical_bernstein":
            raise ValueError("concentration must be 'hoeffding' or 'empirical_bernstein'")
        # Maurer-Pontil (2009), one-sided form. The factor two in the
        # logarithm accounts for the theorem's 2 exp(-x) tail probability.
        x = np.log(2.0 * n_events / alpha)
        variance = float(np.var(values, ddof=1))
        return float(
            np.sqrt(2.0 * variance * x / y.size)
            + 7.0 * value_range * x / (3.0 * (y.size - 1))
        )

    risk_radii = np.asarray([
        radius(values, absolute_error_bound ** 2) for values in empirical_risk_samples
    ])
    lower_risks = np.maximum(empirical_risks - risk_radii, 0.0)
    rows = []
    for i, j in pair_indices:
        disagreement_samples = (p[i] - p[j]) ** 2
        empirical_d = float(np.mean(disagreement_samples))
        disagreement_radius = radius(disagreement_samples, (2.0 * absolute_error_bound) ** 2)
        upper_d = empirical_d + disagreement_radius
        bound = 0.5 * (
            lower_risks[i] + lower_risks[j] - upper_d / (1.0 - lambda_max)
        )
        rows.append({
            "pair": (i, j), "empirical_disagreement": empirical_d,
            "upper_disagreement": float(upper_d),
            "disagreement_radius": float(disagreement_radius),
            "bound": float(max(bound, 0.0)),
        })
    active = max(rows, key=lambda row: row["bound"])
    return {
        "status": "complete", "claim_tier": "conditional",
        "bound": float(active["bound"]), "confidence": float(1.0 - alpha),
        "lambda_max": float(lambda_max), "absolute_error_bound": float(absolute_error_bound),
        "active_pair": active["pair"], "pairs": rows,
        "empirical_risks": empirical_risks.tolist(),
        "lower_risks": lower_risks.tolist(),
        "risk_radii": risk_radii.tolist(),
        "concentration": concentration,
        "assumptions": (
            "squared loss",
            "fixed predictors evaluated on independent identically distributed rows",
            "declared absolute prediction-error cap holds almost surely",
            "every pairwise predictor-error correlation is at most lambda_max",
        ),
    }


def bounded_cluster_pairwise_bound(
    y: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    lambda_max: float,
    absolute_error_bound: float,
    *,
    prediction_difference_bound: float | None = None,
    alpha: float = 0.05,
) -> dict:
    """Case-weighted bound for independent clusters using weighted Hoeffding.

    Each cluster contributes its within-cluster mean and receives weight
    ``n_g / n``. Conditional on the observed cluster sizes, weighted Hoeffding
    gives a simultaneous confidence event. Predictors must be frozen before the
    evaluation clusters are observed; ordinary overlapping OOF predictions do
    not satisfy that requirement.
    """
    y = np.asarray(y, float).ravel()
    p = np.atleast_2d(np.asarray(predictions, float))
    groups = np.asarray(groups).ravel()
    if p.shape[1] != y.size or groups.size != y.size or p.shape[0] < 2:
        raise ValueError("y, predictions, and groups must be aligned with at least two predictors")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("evaluation arrays must be finite")
    if not 0 <= lambda_max < 1:
        raise ValueError("lambda_max must be in [0, 1)")
    if not np.isfinite(absolute_error_bound) or absolute_error_bound <= 0:
        raise ValueError("absolute_error_bound must be finite and positive")
    difference_bound = (
        2.0 * absolute_error_bound
        if prediction_difference_bound is None else float(prediction_difference_bound)
    )
    if not np.isfinite(difference_bound) or difference_bound <= 0:
        raise ValueError("prediction_difference_bound must be finite and positive")
    observed_error = float(np.max(np.abs(p - y[None, :])))
    observed_difference = float(np.max(np.abs(p[:, None, :] - p[None, :, :])))
    if observed_error > absolute_error_bound + 1e-12:
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "observed prediction error exceeds the declared almost-sure cap"}
    if observed_difference > difference_bound + 1e-12:
        return {"status": "refused", "claim_tier": "refused", "bound": None,
                "reason": "observed pairwise prediction difference exceeds the declared cap"}

    unique, inverse, counts = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    if unique.size < 2:
        raise ValueError("cluster concentration requires at least two groups")
    weights = counts.astype(float) / y.size
    sum_weight_squares = float(np.sum(weights ** 2))
    b = p.shape[0]
    pairs = tuple(combinations(range(b), 2))
    n_events = b + len(pairs)
    common_factor = np.sqrt(0.5 * sum_weight_squares * np.log(n_events / alpha))
    risk_radius = absolute_error_bound ** 2 * common_factor
    disagreement_radius = difference_bound ** 2 * common_factor
    risks = np.mean((p - y[None, :]) ** 2, axis=1)
    lower_risks = np.maximum(risks - risk_radius, 0.0)
    rows = []
    for i, j in pairs:
        disagreement = float(np.mean((p[i] - p[j]) ** 2))
        upper = disagreement + disagreement_radius
        pair_bound = 0.5 * (
            lower_risks[i] + lower_risks[j] - upper / (1.0 - lambda_max)
        )
        rows.append({"pair": (i, j), "bound": float(max(pair_bound, 0.0)),
                     "empirical_disagreement": disagreement, "upper_disagreement": upper})
    active = max(rows, key=lambda row: row["bound"])
    return {
        "status": "complete", "claim_tier": "conditional", "bound": active["bound"],
        "confidence": 1.0 - alpha, "lambda_max": float(lambda_max),
        "active_pair": active["pair"], "pairs": rows,
        "n_clusters": int(unique.size),
        "effective_clusters": float(1.0 / sum_weight_squares),
        "max_cluster_size": int(counts.max()),
        "risk_radius": float(risk_radius),
        "disagreement_radius": float(disagreement_radius),
        "assumptions": (
            "squared loss and fixed predictors",
            "evaluation clusters are mutually independent",
            "predictors were frozen before evaluation clusters were observed",
            "declared error and predictor-difference caps hold almost surely",
            "every pairwise predictor-error correlation is at most lambda_max",
        ),
    }


def empirical_bernstein_patient_pairwise_bound(
    y: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    lambda_max: float,
    absolute_error_bound: float,
    prediction_difference_bound: float,
    *,
    alpha: float = 0.05,
    concentration: str = "empirical_bernstein",
) -> dict:
    """Variance-adaptive bound for iid patients with bounded patient means.

    The sampling unit is a patient. Each patient contributes its mean case loss,
    so patients receive equal weight regardless of their number of cases. Unlike
    the Catoni rung, this result needs no externally supplied variance cap; it
    instead needs honest almost-sure ranges for prediction error and pairwise
    prediction differences.
    """
    y = np.asarray(y, float).ravel()
    p = np.atleast_2d(np.asarray(predictions, float))
    groups = np.asarray(groups).ravel()
    if p.shape[1] != y.size or groups.size != y.size or p.shape[0] < 2:
        raise ValueError("y, predictions, and groups must be aligned")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("evaluation arrays must be finite")
    if not 0 <= lambda_max < 1:
        raise ValueError("lambda_max must be in [0, 1)")
    if (not np.isfinite(absolute_error_bound) or absolute_error_bound <= 0 or
            not np.isfinite(prediction_difference_bound) or prediction_difference_bound <= 0):
        raise ValueError("range bounds must be finite and positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if concentration not in {"empirical_bernstein", "hoeffding"}:
        raise ValueError("concentration must be 'empirical_bernstein' or 'hoeffding'")
    observed_error = float(np.max(np.abs(p - y[None, :])))
    observed_difference = float(np.max(np.abs(p[:, None, :] - p[None, :, :])))
    if observed_error > absolute_error_bound + 1e-12:
        return {
            "status": "refused", "claim_tier": "refused", "bound": None,
            "reason": "observed prediction error exceeds the declared almost-sure cap",
            "observed_max_error": observed_error,
        }
    if observed_difference > prediction_difference_bound + 1e-12:
        return {
            "status": "refused", "claim_tier": "refused", "bound": None,
            "reason": "observed pairwise prediction difference exceeds the declared cap",
            "observed_max_difference": observed_difference,
        }

    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    if unique.size < 2:
        raise ValueError("empirical Bernstein concentration requires at least two patients")

    def patient_means(values: np.ndarray) -> np.ndarray:
        return np.bincount(inverse, weights=values, minlength=unique.size) / counts

    b = p.shape[0]
    pairs = tuple(combinations(range(b), 2))
    n_events = b + len(pairs)
    log_term = np.log(2.0 * n_events / alpha)

    def upper_radius(values: np.ndarray, value_range: float) -> float:
        if concentration == "hoeffding":
            return float(value_range * np.sqrt(log_term / (2.0 * unique.size)))
        variance = float(np.var(values, ddof=1))
        return float(
            np.sqrt(2.0 * variance * log_term / unique.size)
            + 7.0 * value_range * log_term / (3.0 * (unique.size - 1))
        )

    risk_values = [patient_means((p[i] - y) ** 2) for i in range(b)]
    risk_estimates = np.asarray([np.mean(values) for values in risk_values])
    risk_radii = np.asarray([
        upper_radius(values, absolute_error_bound ** 2) for values in risk_values
    ])
    lower_risks = np.maximum(risk_estimates - risk_radii, 0.0)
    upper_risks = risk_estimates + risk_radii
    rows = []
    for i, j in pairs:
        values = patient_means((p[i] - p[j]) ** 2)
        estimate = float(np.mean(values))
        radius = upper_radius(values, prediction_difference_bound ** 2)
        upper = estimate + radius
        conservative_bound = 0.5 * (
            lower_risks[i] + lower_risks[j] - upper / (1.0 - lambda_max)
        )
        sharp_bound = _sharp_rectangle_lower_bound(
            lower_risks[i], upper_risks[i], lower_risks[j], upper_risks[j],
            upper, lambda_max,
        )
        bound = conservative_bound if sharp_bound is None else max(conservative_bound, sharp_bound)
        rows.append({
            "pair": (i, j), "bound": float(max(bound, 0.0)),
            "conservative_bound": float(max(conservative_bound, 0.0)),
            "sharp_rectangle_bound": sharp_bound,
            "empirical_disagreement": estimate,
            "upper_disagreement": float(upper),
            "disagreement_radius": float(radius),
        })
    active = max(rows, key=lambda row: row["bound"])
    return {
        "status": "complete", "claim_tier": "conditional",
        "bound": float(active["bound"]), "confidence": 1.0 - alpha,
        "lambda_max": float(lambda_max), "active_pair": active["pair"],
        "pairs": rows, "n_patients": int(unique.size), "n_events": n_events,
        "estimand": "equal-patient expected mean squared case error",
        "empirical_risks": risk_estimates.tolist(),
        "lower_risks": lower_risks.tolist(), "upper_risks": upper_risks.tolist(),
        "risk_radii": risk_radii.tolist(),
        "absolute_error_bound": float(absolute_error_bound),
        "prediction_difference_bound": float(prediction_difference_bound),
        "concentration": f"patient-level {concentration.replace('_', ' ')}",
        "moment_geometry": "sharp pairwise formula minimised over simultaneous confidence rectangles",
        "assumptions": (
            "squared loss and fixed predictors",
            "patients are independent and identically distributed sampling units",
            "predictors were frozen before evaluation patients were observed",
            "declared error and predictor-difference caps hold almost surely",
            "every pairwise predictor-error correlation is at most lambda_max",
        ),
    }


def bounded_cluster_sensitivity_curve(
    y: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    lambdas: np.ndarray,
    absolute_error_bound: float,
    *,
    prediction_difference_bound: float | None = None,
    alpha: float = 0.05,
) -> dict:
    """Simultaneous lambda sensitivity from one shared confidence event."""
    grid = np.asarray(lambdas, float)
    if grid.ndim != 1 or grid.size == 0 or np.any(np.diff(grid) < 0):
        raise ValueError("lambdas must be a nonempty sorted vector")
    rows = [
        bounded_cluster_pairwise_bound(
            y, predictions, groups, float(value), absolute_error_bound,
            prediction_difference_bound=prediction_difference_bound, alpha=alpha,
        )
        for value in grid
    ]
    if any(row["status"] != "complete" for row in rows):
        refusal = next(row for row in rows if row["status"] != "complete")
        return {"status": "refused", "claim_tier": "refused", "reason": refusal["reason"]}
    bounds = np.asarray([row["bound"] for row in rows])
    return {
        "status": "complete", "claim_tier": "conditional",
        "confidence": 1.0 - alpha,
        "simultaneous_over_lambda_grid": True,
        "curve": [
            {"lambda_max": float(value), "bound": float(bound)}
            for value, bound in zip(grid, bounds)
        ],
        "diagnostics": {
            "n_clusters": rows[0]["n_clusters"],
            "effective_clusters": rows[0]["effective_clusters"],
            "risk_radius": rows[0]["risk_radius"],
            "disagreement_radius": rows[0]["disagreement_radius"],
        },
        "assumptions": rows[0]["assumptions"],
    }


def median_of_means_cluster_pairwise_bound(
    y: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    lambda_max: float,
    risk_variance_bound: float,
    disagreement_variance_bound: float,
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Heavy-tail patient-level bound using median-of-means moments.

    The estimand weights patients equally: each patient's contribution is its
    mean case loss. Only finite, predeclared upper bounds on the variances of
    those patient-level risk and disagreement variables are required. No
    bounded outcome or loss is assumed.
    """
    y = np.asarray(y, float).ravel()
    p = np.atleast_2d(np.asarray(predictions, float))
    groups = np.asarray(groups).ravel()
    if p.shape[1] != y.size or groups.size != y.size or p.shape[0] < 2:
        raise ValueError("y, predictions, and groups must be aligned")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("evaluation arrays must be finite")
    if not 0 <= lambda_max < 1:
        raise ValueError("lambda_max must be in [0, 1)")
    if risk_variance_bound <= 0 or disagreement_variance_bound <= 0:
        raise ValueError("variance bounds must be positive")
    if not np.isfinite([risk_variance_bound, disagreement_variance_bound]).all():
        raise ValueError("variance bounds must be finite")

    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    b = p.shape[0]
    pairs = tuple(combinations(range(b), 2))
    n_events = b + len(pairs)
    n_blocks = int(np.ceil(8.0 * np.log(n_events / alpha)))
    block_size = int(unique.size // n_blocks)
    if block_size < 2:
        raise ValueError("too few independent clusters for simultaneous median-of-means bounds")
    order = np.random.default_rng(seed).permutation(unique.size)[:n_blocks * block_size]
    blocks = order.reshape(n_blocks, block_size)

    def group_means(values: np.ndarray) -> np.ndarray:
        return np.bincount(inverse, weights=values, minlength=unique.size) / counts

    def mom(values: np.ndarray) -> float:
        return float(np.median(np.mean(values[blocks], axis=1)))

    risk_radius = 2.0 * np.sqrt(risk_variance_bound / block_size)
    disagreement_radius = 2.0 * np.sqrt(disagreement_variance_bound / block_size)
    risk_mom = np.asarray([mom(group_means((p[i] - y) ** 2)) for i in range(b)])
    lower_risks = np.maximum(risk_mom - risk_radius, 0.0)
    rows = []
    for i, j in pairs:
        d_mom = mom(group_means((p[i] - p[j]) ** 2))
        upper_d = d_mom + disagreement_radius
        bound = 0.5 * (
            lower_risks[i] + lower_risks[j] - upper_d / (1.0 - lambda_max)
        )
        rows.append({"pair": (i, j), "bound": float(max(bound, 0.0)),
                     "mom_disagreement": d_mom, "upper_disagreement": upper_d})
    active = max(rows, key=lambda row: row["bound"])
    return {
        "status": "complete", "claim_tier": "conditional", "bound": active["bound"],
        "confidence": 1.0 - alpha, "lambda_max": float(lambda_max),
        "estimand": "equal-patient expected mean squared case error",
        "active_pair": active["pair"], "pairs": rows,
        "n_clusters": int(unique.size), "n_blocks": n_blocks,
        "block_size": block_size, "discarded_clusters": int(unique.size - n_blocks * block_size),
        "risk_mom": risk_mom.tolist(), "lower_risks": lower_risks.tolist(),
        "risk_radius": float(risk_radius),
        "disagreement_radius": float(disagreement_radius),
        "assumptions": (
            "squared loss and fixed predictors",
            "patients are independent and identically distributed sampling units",
            "predictors were frozen before evaluation patients were observed",
            "patient-level risk variance is at most risk_variance_bound",
            "patient-level disagreement variance is at most disagreement_variance_bound",
            "every pairwise predictor-error correlation is at most lambda_max",
        ),
    }


def _catoni_mean(values: np.ndarray, variance_bound: float, delta: float) -> tuple[float, float, float]:
    """Catoni (2012) location estimate and its two-sided radius."""
    values = np.asarray(values, float).ravel()
    n = values.size
    log_term = np.log(1.0 / delta)
    if n <= 2.0 * log_term:
        raise ValueError("too few clusters for the requested Catoni confidence level")
    radius = np.sqrt(2.0 * variance_bound * log_term / (n - 2.0 * log_term))
    tuning = np.sqrt(
        2.0 * log_term
        / (n * variance_bound * (1.0 + 2.0 * log_term / (n - 2.0 * log_term)))
    )

    def psi(x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            np.log1p(x + 0.5 * x * x),
            -np.log1p(-x + 0.5 * x * x),
        )

    def score(location: float) -> float:
        return float(np.sum(psi(tuning * (values - location))))

    lo, hi = float(np.min(values)), float(np.max(values))
    estimate = lo if lo == hi else float(brentq(score, lo, hi, xtol=1e-12, rtol=1e-12))
    return estimate, float(radius), float(tuning)


def catoni_cluster_pairwise_bound(
    y: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    lambda_max: float,
    risk_variance_bound: float,
    disagreement_variance_bound: float,
    *,
    alpha: float = 0.05,
) -> dict:
    """Heavy-tail patient-level dependence bound with Catoni moments.

    The guarantee needs only predeclared finite variance upper bounds for the
    patient-level risk and disagreement variables. Confidence is simultaneous
    across all moments by a union bound.
    """
    y = np.asarray(y, float).ravel()
    p = np.atleast_2d(np.asarray(predictions, float))
    groups = np.asarray(groups).ravel()
    if p.shape[1] != y.size or groups.size != y.size or p.shape[0] < 2:
        raise ValueError("y, predictions, and groups must be aligned")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("evaluation arrays must be finite")
    if not 0 <= lambda_max < 1:
        raise ValueError("lambda_max must be in [0, 1)")
    if not np.isfinite([risk_variance_bound, disagreement_variance_bound]).all():
        raise ValueError("variance bounds must be finite")
    if risk_variance_bound <= 0 or disagreement_variance_bound <= 0:
        raise ValueError("variance bounds must be positive")

    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    if unique.size < 3:
        raise ValueError("Catoni concentration requires at least three independent clusters")

    def group_means(values: np.ndarray) -> np.ndarray:
        return np.bincount(inverse, weights=values, minlength=unique.size) / counts

    b = p.shape[0]
    pairs = tuple(combinations(range(b), 2))
    n_events = b + len(pairs)
    # Catoni's stated two-sided failure is 2*delta. Allocate alpha/K.
    event_delta = alpha / (2.0 * n_events)
    risk_rows = [
        _catoni_mean(group_means((p[i] - y) ** 2), risk_variance_bound, event_delta)
        for i in range(b)
    ]
    risk_estimates = np.asarray([row[0] for row in risk_rows])
    risk_radius = risk_rows[0][1]
    lower_risks = np.maximum(risk_estimates - risk_radius, 0.0)
    rows = []
    for i, j in pairs:
        estimate, radius, _ = _catoni_mean(
            group_means((p[i] - p[j]) ** 2), disagreement_variance_bound, event_delta
        )
        upper = estimate + radius
        bound = 0.5 * (
            lower_risks[i] + lower_risks[j] - upper / (1.0 - lambda_max)
        )
        rows.append({"pair": (i, j), "bound": float(max(bound, 0.0)),
                     "catoni_disagreement": estimate, "upper_disagreement": upper})
    active = max(rows, key=lambda row: row["bound"])
    return {
        "status": "complete", "claim_tier": "conditional", "bound": active["bound"],
        "confidence": 1.0 - alpha, "lambda_max": float(lambda_max),
        "estimand": "equal-patient expected mean squared case error",
        "active_pair": active["pair"], "pairs": rows,
        "n_clusters": int(unique.size), "n_events": n_events,
        "catoni_risks": risk_estimates.tolist(), "lower_risks": lower_risks.tolist(),
        "risk_radius": float(risk_radius),
        "disagreement_radius": float(rows[0]["upper_disagreement"] - rows[0]["catoni_disagreement"]),
        "risk_variance_bound": float(risk_variance_bound),
        "disagreement_variance_bound": float(disagreement_variance_bound),
        "assumptions": (
            "squared loss and fixed predictors",
            "patients are independent and identically distributed sampling units",
            "predictors were frozen before evaluation patients were observed",
            "patient-level risk variance is at most risk_variance_bound",
            "patient-level disagreement variance is at most disagreement_variance_bound",
            "every pairwise predictor-error correlation is at most lambda_max",
        ),
    }
