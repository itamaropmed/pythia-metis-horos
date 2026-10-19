"""Known-risk test of covariate-only 2-opt matching on the hard sparse design."""
from __future__ import annotations

import numpy as np

from horos.represent.pairing import distance_refined_pairs, global_nearest_pairs
from pythia.core.replicates import paired_replicate_lower_bound

from .multiloss_comparison import LOSSES
from .multiloss_synthetic import multiloss_design


def run_sparse_pair_refinement(
    *, seed: int, repetitions: int = 10, n_entities: int = 512,
) -> dict:
    if repetitions < 1 or n_entities < 32 or n_entities % 2:
        raise ValueError("repetitions >=1 and even n_entities >=32 required")
    seeds = np.random.SeedSequence(seed).spawn(repetitions)
    rows = []
    for generator in seeds:
        design = multiloss_design(
            np.random.default_rng(generator), "numeric_sparse4", n_entities,
        )
        x, y = design["representation"], design["outcomes"]
        entities = np.arange(n_entities)
        old = global_nearest_pairs(x, entities)
        new = distance_refined_pairs(x, entities, max_passes=3)
        for label, loss, quantile, window in LOSSES:
            def bound(pairs):
                return paired_replicate_lower_bound(
                    y[pairs.left_indices], y[pairs.right_indices],
                    pairs.distances, entities[pairs.left_indices],
                    entities[pairs.right_indices], pairs.pair_blocks,
                    loss=loss, quantile=quantile, window=window,
                    lipschitz_bound=design["lipschitz_cap"],
                    distance_bound=1.,
                    outcome_difference_bound=300. if loss == "pinball" else None,
                    alpha=.01, concentration="conditional_bentkus",
                )["bound"]
            rows.append({
                "loss": label, "truth": float(np.mean(design[f"truth_{label}"])),
                "old": bound(old), "new": bound(new),
                "old_distance": float(np.mean(old.distances)),
                "new_distance": float(np.mean(new.distances)),
            })
    output = {}
    for label, _, _, _ in LOSSES:
        subset = [row for row in rows if row["loss"] == label]
        old = np.array([row["old"] for row in subset])
        new = np.array([row["new"] for row in subset])
        truth = np.array([row["truth"] for row in subset])
        output[label] = {
            "new_valid": int(np.sum(new <= truth + 1e-12)),
            "new_strict_wins": int(np.sum(new > old + 1e-10)),
            "ties": int(np.sum(np.abs(new - old) <= 1e-10)),
            "new_strict_losses": int(np.sum(new + 1e-10 < old)),
            "old_median_bound": float(np.median(old)),
            "new_median_bound": float(np.median(new)),
        }
    return {
        "seed": seed, "repetitions": repetitions,
        "old_mean_pair_distance": float(np.mean([row["old_distance"] for row in rows])),
        "new_mean_pair_distance": float(np.mean([row["new_distance"] for row in rows])),
        "pythia_trained_predictive_model": False,
        "by_loss": output,
    }
