"""Small-n global constrained-omniscient MAE comparator.

Inspired by Anderson--Bjarnadottir (2024), not a reproduction of their
particular threshold/link-selection protocol. Uses all pairs and a supplied
Lipschitz cap so the smoothness assumption matches Pythia on each DGP.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


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
    edges = len(i)
    # Variables are p[0:n] (free) and residual t[0:n] (nonnegative).
    # 2n residual rows plus 2*edges smoothness rows.
    row = np.concatenate([
        np.arange(n), np.arange(n),
        n + np.arange(n), n + np.arange(n),
        2 * n + np.arange(edges), 2 * n + np.arange(edges),
        2 * n + edges + np.arange(edges), 2 * n + edges + np.arange(edges),
    ])
    col = np.concatenate([
        np.arange(n), n + np.arange(n),
        np.arange(n), n + np.arange(n),
        i, j, i, j,
    ])
    values = np.concatenate([
        np.ones(n), -np.ones(n),
        -np.ones(n), -np.ones(n),
        np.ones(edges), -np.ones(edges),
        -np.ones(edges), np.ones(edges),
    ])
    constraint = coo_matrix((values, (row, col)), shape=(2 * n + 2 * edges, 2 * n))
    rhs = np.concatenate([
        y, -y,
        lipschitz_bound * distances, lipschitz_bound * distances,
    ])
    objective = np.concatenate([np.zeros(n), np.ones(n) / n])
    fit = linprog(
        objective, A_ub=constraint.tocsr(), b_ub=rhs,
        bounds=[(None, None)] * n + [(0, None)] * n,
        method="highs",
    )
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
