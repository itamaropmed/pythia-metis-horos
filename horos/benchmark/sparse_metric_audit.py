"""Paired known-truth audit of a predeclared signal-aligned metric.

The generator states the relevant coordinate before outcomes are sampled.
This is an oracle scientific control, not automatic feature discovery.
"""
from __future__ import annotations

import numpy as np

from horos.represent.pairing import global_nearest_pairs
from pythia.core.replicates import paired_replicate_lower_bound

from .dependence_comparison import _wilson_lower
from .replicate_adversarial import _design


def run_sparse_metric_audit(
    *, seed: int = 20260912, repetitions: int = 20,
    entity_counts: tuple[int, ...] = (1000, 4000, 10000),
) -> dict:
    """Compare mean-L1 over ten features with a frozen one-feature metric."""
    seeds = np.random.SeedSequence(seed).spawn(len(entity_counts) * repetitions)
    rows = []
    cursor = 0
    for n in entity_counts:
        for _ in range(repetitions):
            rng = np.random.default_rng(seeds[cursor]); cursor += 1
            x, y, full_l, risks, cap, _ = _design(
                rng, "sparse_ten_dimensional", n
            )
            entities = np.arange(n)
            designs = {
                "all_ten_equal_weight": (x, full_l),
                "predeclared_first_coordinate": (x[:, :1], 3.0),
            }
            for metric_name, (metric, l_cap) in designs.items():
                pairs = global_nearest_pairs(metric, entities)
                for loss in ("mae", "mse"):
                    result = paired_replicate_lower_bound(
                        y[pairs.left_indices], y[pairs.right_indices],
                        pairs.distances, entities[pairs.left_indices],
                        entities[pairs.right_indices], pairs.pair_blocks,
                        loss=loss, lipschitz_bound=l_cap,
                        distance_bound=1.0, outcome_difference_bound=cap,
                        concentration="conditional_empirical_bernstein",
                    )
                    rows.append({
                        "n": n, "loss": loss, "metric": metric_name,
                        "bound": result["bound"], "estimate": result["estimate"],
                        "true_risk": risks[loss],
                        "mean_pair_distance": result["mean_distance"],
                    })

    def summarize(selected):
        bound = np.asarray([row["bound"] for row in selected])
        estimate = np.asarray([row["estimate"] for row in selected])
        truth = np.asarray([row["true_risk"] for row in selected])
        valid = bound <= truth + 1e-12
        return {
            "valid": int(valid.sum()), "total": int(valid.size),
            "wilson_lower_95": _wilson_lower(int(valid.sum()), int(valid.size)),
            "median_bound": float(np.median(bound)),
            "median_tightness": float(np.median(np.where(valid, bound / truth, 0))),
            "median_estimator_absolute_error": float(np.median(np.abs(estimate - truth))),
            "median_pair_distance": float(np.median([
                row["mean_pair_distance"] for row in selected
            ])),
        }

    return {
        "status": "exploratory_known_support_control",
        "pythia_trained_models": False, "seed": seed,
        "repetitions": repetitions,
        "n_generated_datasets": len(entity_counts) * repetitions,
        "metric_selected_from_outcomes": False,
        "metric_selection_note": (
            "the relevant first coordinate is declared by the generator; "
            "this is not an automatic real-data metric-selection algorithm"
        ),
        "results": {
            loss: {
                metric: summarize([
                    row for row in rows if row["loss"] == loss and row["metric"] == metric
                ])
                for metric in ("all_ten_equal_weight", "predeclared_first_coordinate")
            }
            for loss in ("mae", "mse")
        },
    }
