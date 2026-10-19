"""Three established out-of-fold local-action baselines, outside Pythia."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, tau: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, tau * cumulative[-1], side="left")])


def _weighted_window_center(values: np.ndarray, weights: np.ndarray, width: float) -> float:
    order = np.argsort(values, kind="stable")
    v, mass = values[order], weights[order]
    left = 0
    running = 0.
    best = (-1., 0, 0)
    for right in range(len(v)):
        running += mass[right]
        while v[right] - v[left] > 2 * width:
            running -= mass[left]
            left += 1
        if running > best[0]:
            best = (running, left, right)
    return float(.5 * (v[best[1]] + v[best[2]]))


def _histogram_key(row: np.ndarray) -> tuple[int, ...]:
    first = min(7, int(np.floor(row[0] * 8)))
    return (first,) if row.size == 1 else (first, int(round(row[1] * 2)))


def crossfit_local_decision_plugins(
    representation: np.ndarray,
    outcomes: np.ndarray,
    *,
    seed: int = 31,
    neighbors: int = 51,
    bandwidth: float = .12,
) -> dict:
    """Cross-fit kNN, fixed histogram, and Gaussian-kernel plug-in actions.

    These are fitted predictive rules and therefore live only in Horos. Their
    achieved risks are point-estimation baselines/upper performance, never
    lower confidence bounds on Bayes risk.
    """
    x = np.asarray(representation, float)
    y = np.asarray(outcomes, float).ravel()
    if (x.ndim != 2 or len(x) != len(y) or len(y) < 20
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or neighbors < 2 or bandwidth <= 0):
        raise ValueError("aligned finite representation and valid fixed smoothing parameters required")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(y))
    folds = np.array_split(permutation, 2)
    actions = {
        method: {"q10": np.empty(len(y)), "q90": np.empty(len(y)),
                 **{f"w{width}": np.empty(len(y)) for width in (15, 30, 60)}}
        for method in ("knn", "histogram", "kernel")
    }
    for eval_rows, train_rows in ((folds[0], folds[1]), (folds[1], folds[0])):
        train_x, train_y = x[train_rows], y[train_rows]
        tree = cKDTree(train_x)
        k = min(neighbors, len(train_rows))
        kernel_k = min(max(100, neighbors), len(train_rows))
        _, knn_indices = tree.query(x[eval_rows], k=k, p=1)
        kernel_distances, kernel_indices = tree.query(x[eval_rows], k=kernel_k, p=1)
        bins: dict[tuple[int, ...], list[int]] = {}
        for index, row in enumerate(train_x):
            bins.setdefault(_histogram_key(row), []).append(index)
        for position, row_id in enumerate(eval_rows):
            candidates = np.atleast_1d(knn_indices[position])
            sample_sets = {
                "knn": (train_y[candidates], np.ones(len(candidates))),
            }
            histogram_indices = bins.get(_histogram_key(x[row_id]), [])
            if len(histogram_indices) < 5:
                histogram_indices = list(range(len(train_y)))
            sample_sets["histogram"] = (
                train_y[histogram_indices], np.ones(len(histogram_indices)),
            )
            indices = np.atleast_1d(kernel_indices[position])
            distances = np.atleast_1d(kernel_distances[position]) / x.shape[1]
            weights = np.exp(-.5 * (distances / bandwidth) ** 2)
            sample_sets["kernel"] = (train_y[indices], weights)
            for method, (values, weights) in sample_sets.items():
                actions[method]["q10"][row_id] = _weighted_quantile(values, weights, .1)
                actions[method]["q90"][row_id] = _weighted_quantile(values, weights, .9)
                for width in (15, 30, 60):
                    actions[method][f"w{width}"][row_id] = _weighted_window_center(
                        values, weights, float(width),
                    )
    results = {}
    for method, values in actions.items():
        results[method] = {
            "q10_pinball": float(np.mean(np.maximum(
                .1 * (y - values["q10"]), -.9 * (y - values["q10"]),
            ))),
            "q90_pinball": float(np.mean(np.maximum(
                .9 * (y - values["q90"]), -.1 * (y - values["q90"]),
            ))),
            **{
                f"w{width}_miss": float(np.mean(
                    np.abs(y - values[f"w{width}"]) > width,
                )) for width in (15, 30, 60)
            },
            **{
                f"w{width}_percentage": float(100 * np.mean(
                    np.abs(y - values[f"w{width}"]) <= width,
                )) for width in (15, 30, 60)
            },
        }
    return {
        "methods": results,
        "pythia_trained_model": False,
        "external_predictive_rules_fitted": True,
        "crossfit_folds": 2,
        "status": "exploratory_achieved_risk_plugins_not_bayes_bounds",
    }
