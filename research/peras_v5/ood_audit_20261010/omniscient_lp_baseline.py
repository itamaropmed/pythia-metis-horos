"""Small-n constrained-omniscient MAE comparators.

``anderson_bjarnadottir_mae_lp`` implements the published default protocol in
Anderson & Bjarnadottir (PLOS ONE, 2024): standardized Euclidean distance,
an outcome-fitted maximum absolute linear-regression coefficient as the link,
and the smallest distance threshold containing at least 5% of all possible
pair constraints and an average of five constraints per observation. It is a
faithful algorithmic comparator, not a confidence bound: its link and graph
use the evaluation outcomes and the paper describes the procedure as a
heuristic without a finite-sample coverage guarantee.

``global_lipschitz_mae_lp`` is the older oracle-cap control. It uses all pairs
and a supplied Lipschitz cap so the smoothness assumption matches Peras on each
synthetic DGP; it must not be labelled as the published protocol.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _solve_mae_lp(outcomes, left, right, allowances):
    """Solve the common MAE LP for a fixed, already selected edge set."""
    y = np.asarray(outcomes, float).ravel()
    left = np.asarray(left, int).ravel()
    right = np.asarray(right, int).ravel()
    allowances = np.asarray(allowances, float).ravel()
    if left.shape != right.shape or left.shape != allowances.shape:
        raise ValueError("edge arrays and allowances must align")
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= y.size) or np.any(right >= y.size):
        raise ValueError("edge indices are out of range")
    if not np.isfinite(allowances).all() or np.any(allowances < 0):
        raise ValueError("edge allowances must be finite and nonnegative")
    n, edges = y.size, left.size
    row = np.concatenate([
        np.arange(n), np.arange(n), n + np.arange(n), n + np.arange(n),
        2*n + np.arange(edges), 2*n + np.arange(edges),
        2*n + edges + np.arange(edges), 2*n + edges + np.arange(edges),
    ])
    col = np.concatenate([
        np.arange(n), n + np.arange(n), np.arange(n), n + np.arange(n),
        left, right, left, right,
    ])
    values = np.concatenate([
        np.ones(n), -np.ones(n), -np.ones(n), -np.ones(n),
        np.ones(edges), -np.ones(edges), -np.ones(edges), np.ones(edges),
    ])
    constraint = coo_matrix((values, (row, col)), shape=(2*n + 2*edges, 2*n))
    rhs = np.concatenate([y, -y, allowances, allowances])
    objective = np.concatenate([np.zeros(n), np.ones(n) / n])
    fit = linprog(
        objective, A_ub=constraint.tocsr(), b_ub=rhs,
        bounds=[(None, None)] * n + [(0, None)] * n, method="highs",
    )
    return fit, int(edges)


def anderson_bjarnadottir_mae_lp(
    representation: np.ndarray,
    outcomes: np.ndarray,
    *,
    pair_fraction: float = 0.05,
    minimum_average_degree: float = 5.0,
    max_entities: int = 250,
) -> dict:
    """Run the published Anderson--Bjarnadottir default MAE protocol.

    The regression-derived link is intentionally estimated on the same
    outcomes optimized by the LP, matching the paper. That label adaptivity is
    why this is an exploratory external heuristic, never a Peras certificate.
    """
    x = np.asarray(representation, float)
    y = np.asarray(outcomes, float).ravel()
    if x.ndim != 2 or x.shape[0] != y.size or y.size < 2:
        raise ValueError("representation and outcomes must align")
    if y.size > max_entities:
        raise ValueError("dense published baseline is restricted to small audits")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("LP inputs must be finite")
    if not 0 < pair_fraction <= 1:
        raise ValueError("pair_fraction must lie in (0, 1]")
    if not np.isfinite(minimum_average_degree) or minimum_average_degree < 0:
        raise ValueError("minimum_average_degree must be finite and nonnegative")

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    active = scale > np.finfo(float).eps
    z = np.zeros_like(x, dtype=float)
    z[:, active] = (x[:, active] - mean[active]) / scale[active]

    design = np.column_stack([np.ones(y.size), z[:, active]])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0][1:]
    beta_max = float(np.max(np.abs(coefficients), initial=0.0))

    left_all, right_all = np.triu_indices(y.size, 1)
    distances_all = np.linalg.norm(z[left_all] - z[right_all], axis=1)
    possible = distances_all.size
    required = max(
        int(np.ceil(pair_fraction * possible)),
        int(np.ceil(minimum_average_degree * y.size / 2.0)),
    )
    required = min(possible, max(1, required))
    order = np.argsort(distances_all, kind="mergesort")
    delta = float(distances_all[order[required - 1]])
    tolerance = 16 * np.finfo(float).eps * max(1.0, abs(delta))
    selected = distances_all <= delta + tolerance
    left, right = left_all[selected], right_all[selected]
    distances = distances_all[selected]
    fit, edges = _solve_mae_lp(y, left, right, beta_max * distances)
    if not fit.success:
        return {"status": "refused", "reason": str(fit.message)}
    return {
        "status": "complete",
        "empirical_omniscient_mae": float(fit.fun),
        "n_entities": int(y.size),
        "n_active_features": int(np.sum(active)),
        "n_pair_constraints": edges,
        "n_possible_pair_constraints": int(possible),
        "pair_fraction_realized": float(edges / possible),
        "average_degree_realized": float(2 * edges / y.size),
        "distance": "standardized_l2",
        "distance_threshold": delta,
        "link_slope_beta_max": beta_max,
        "link_rule": "max_abs_ols_coefficient_times_standardized_l2_distance",
        "protocol_reference": (
            "Anderson and Bjarnadottir (2024), PLOS ONE, "
            "doi:10.1371/journal.pone.0296904"
        ),
        "inferential_status": (
            "published heuristic point floor; label-adaptive and not a "
            "finite-sample confidence bound"
        ),
    }


def global_lipschitz_mae_lp(
    representation: np.ndarray, outcomes: np.ndarray, *,
    lipschitz_bound: float, max_entities: int = 250,
) -> dict:
    """Minimize realized MAE over Lipschitz-constrained omniscient actions."""
    x = np.asarray(representation, float)
    y = np.asarray(outcomes, float).ravel()
    if x.ndim != 2 or x.shape[0] != y.size or y.size < 2:
        raise ValueError("representation and outcomes must align")
    if y.size > max_entities:
        raise ValueError("dense global LP baseline is restricted to small audits")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("LP inputs must be finite")
    if not np.isfinite(lipschitz_bound) or lipschitz_bound < 0:
        raise ValueError("Lipschitz cap must be nonnegative and finite")
    n = y.size
    i, j = np.triu_indices(n, 1)
    distances = np.mean(np.abs(x[i] - x[j]), axis=1)
    fit, edges = _solve_mae_lp(y, i, j, lipschitz_bound * distances)
    if not fit.success:
        return {"status": "refused", "reason": str(fit.message)}
    return {
        "status": "complete", "empirical_omniscient_mae": float(fit.fun),
        "n_entities": int(n), "n_pair_constraints": int(edges),
        "lipschitz_bound": float(lipschitz_bound),
        "inferential_status": "heuristic point witness, not a 95% lower confidence bound",
    }


def jackknife_lipschitz_mae_lp(
    representation: np.ndarray, outcomes: np.ndarray, *,
    lipschitz_bound: float, split_seed: int = 0,
) -> dict:
    """Exploratory half-sample bias correction of the omniscient LP center.

    The split is outcome-blind. This is a point-estimator hypothesis, not a
    theorem or a lower confidence bound; high-dimensional failures are likely.
    """
    x = np.asarray(representation, float)
    y = np.asarray(outcomes, float).ravel()
    if len(y) < 4 or len(y) % 2 or x.shape[0] != len(y):
        raise ValueError("jackknife requires even aligned sample of >=4")
    full = global_lipschitz_mae_lp(x, y, lipschitz_bound=lipschitz_bound)
    indices = np.random.default_rng(split_seed).permutation(len(y))
    halves = [global_lipschitz_mae_lp(
        x[rows], y[rows], lipschitz_bound=lipschitz_bound,
    ) for rows in np.array_split(indices, 2)]
    if any(row["status"] != "complete" for row in (full, *halves)):
        return {"status": "refused", "reason": "LP failed in full or half sample"}
    estimate = max(0.0, 2 * full["empirical_omniscient_mae"] - .5 * sum(
        row["empirical_omniscient_mae"] for row in halves
    ))
    return {
        "status": "complete", "estimate": float(estimate),
        "full_lp": full["empirical_omniscient_mae"],
        "half_lp_mean": float(np.mean([
            row["empirical_omniscient_mae"] for row in halves
        ])),
        "inferential_status": "exploratory bias-corrected point estimate, not a confidence bound",
    }
