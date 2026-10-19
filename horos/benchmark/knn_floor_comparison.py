"""Known-risk point-estimator comparison with external LOO kNN baselines.

The neighbor outcome-based estimator lives in Horos, never inside Pythia.
It is a simple kNN baseline, not an implementation of a cited paper's exact
mutual-neighbor or fused-lasso algorithm.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from horos.represent.pairing import global_nearest_pairs
from pythia.core.replicates import paired_replicate_lower_bound

from .replicate_adversarial import DESIGN_NAMES, _design


def run_knn_floor_comparison(
    *, seed: int = 20260912, repetitions: int = 20,
    n_entities: int = 2000, k: int = 20,
) -> dict:
    """Compare Pythia's lower-bound center with a LOO local-risk plug-in."""
    if n_entities < max(k + 2, 4) or n_entities % 2 or k < 2:
        raise ValueError("n_entities must be even and exceed the neighbor count")
    seeds = np.random.SeedSequence(seed).spawn(len(DESIGN_NAMES) * repetitions)
    rows = []
    cursor = 0
    for name in DESIGN_NAMES:
        for _ in range(repetitions):
            rng = np.random.default_rng(seeds[cursor]); cursor += 1
            x, y, l_cap, truths, cap, _ = _design(rng, name, n_entities)
            entities = np.arange(n_entities)
            design = global_nearest_pairs(x, entities)
            _, neighbor_indices = cKDTree(x).query(x, k=k + 1, p=1)
            # Query rows may not be first under ties; exclude them explicitly.
            neighbors = np.asarray([
                row[row != i][:k] for i, row in enumerate(neighbor_indices)
            ])
            if neighbors.shape != (n_entities, k):
                raise AssertionError("kNN leave-one-out neighbor count changed")
            neighbor_y = y[neighbors]
            for loss in ("mae", "mse"):
                pair = paired_replicate_lower_bound(
                    y[design.left_indices], y[design.right_indices],
                    design.distances, entities[design.left_indices],
                    entities[design.right_indices], design.pair_blocks,
                    loss=loss, lipschitz_bound=l_cap,
                    distance_bound=1.0, outcome_difference_bound=cap,
                    concentration="conditional_empirical_bernstein",
                )
                location = (
                    np.median(neighbor_y, axis=1) if loss == "mae"
                    else np.mean(neighbor_y, axis=1)
                )
                residual = np.abs(y - location)
                knn_estimate = float(np.mean(
                    residual if loss == "mae" else residual ** 2
                ))
                rows.append({
                    "design": name, "loss": loss, "truth": truths[loss],
                    "pythia_center": pair["estimate"],
                    "pythia_bound": pair["bound"],
                    "knn_loo_risk": knn_estimate,
                })

    def summarize(selected):
        truth = np.asarray([row["truth"] for row in selected])
        pythia_error = np.abs(np.asarray([row["pythia_center"] for row in selected]) - truth)
        knn_error = np.abs(np.asarray([row["knn_loo_risk"] for row in selected]) - truth)
        pythia_wins = int(np.sum(pythia_error + 1e-12 < knn_error))
        knn_wins = int(np.sum(knn_error + 1e-12 < pythia_error))
        return {
            "runs": len(selected),
            "median_pythia_center_absolute_error": float(np.median(pythia_error)),
            "median_knn_loo_absolute_error": float(np.median(knn_error)),
            "pythia_strict_wins": pythia_wins, "knn_strict_wins": knn_wins,
            "ties": len(selected) - pythia_wins - knn_wins,
        }

    return {
        "status": "exploratory_external_point_estimator_comparison",
        "pythia_trained_models": False,
        "horos_knn_uses_outcomes": True,
        "knn_is_a_bound": False,
        "same_estimand_warning": (
            "Pythia center is a lower-bound plug-in; leave-one-out kNN risk "
            "is an achieved-risk plug-in, not a confidence lower bound"
        ),
        "seed": seed, "n_entities": n_entities, "k": k,
        "repetitions_per_design": repetitions,
        "summary": summarize(rows),
        "by_loss": {
            loss: summarize([row for row in rows if row["loss"] == loss])
            for loss in ("mae", "mse")
        },
        "by_design": {
            name: summarize([row for row in rows if row["design"] == name])
            for name in DESIGN_NAMES
        },
    }
