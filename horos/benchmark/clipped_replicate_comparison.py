"""Known-risk comparison of clipped and unclipped paired confidence bounds."""
from __future__ import annotations

import numpy as np

from horos.represent.pairing import global_nearest_pairs
from pythia.core.replicates import (
    paired_replicate_clipped_envelope, paired_replicate_lower_bound,
)

from .dependence_comparison import _wilson_lower
from .replicate_adversarial import DESIGN_NAMES, _design


def run_clipped_replicate_comparison(
    *, seed: int = 20260918, repetitions: int = 20,
    n_entities: int = 2000,
) -> dict:
    """Test a dimensionless predeclared clip grid on ten known-risk DGPs."""
    if n_entities < 4 or n_entities % 2 or repetitions < 1:
        raise ValueError("n_entities must be even and repetitions positive")
    seeds = np.random.SeedSequence(seed).spawn(len(DESIGN_NAMES) * repetitions)
    rows = []
    cursor = 0
    for name in DESIGN_NAMES:
        for _ in range(repetitions):
            rng = np.random.default_rng(seeds[cursor]); cursor += 1
            x, y, l_cap, true_risks, difference_cap, _ = _design(
                rng, name, n_entities
            )
            entities = np.arange(n_entities)
            pairs = global_nearest_pairs(x, entities)
            for loss in ("mae", "mse"):
                term_cap = (
                    .5 * difference_cap if loss == "mae"
                    else .5 * difference_cap ** 2
                )
                kwargs = dict(
                    loss=loss, lipschitz_bound=l_cap,
                    distance_bound=1.0,
                    outcome_difference_bound=difference_cap,
                    concentration="conditional_empirical_bernstein",
                )
                args = (
                    y[pairs.left_indices], y[pairs.right_indices],
                    pairs.distances, entities[pairs.left_indices],
                    entities[pairs.right_indices], pairs.pair_blocks,
                )
                plain = paired_replicate_lower_bound(*args, **kwargs)
                clipped = paired_replicate_clipped_envelope(
                    *args, **kwargs,
                    outcome_term_clips=tuple(
                        term_cap * ratio for ratio in (.125, .25, .5, 1.0)
                    ),
                )
                rows.append({
                    "design": name, "loss": loss, "truth": true_risks[loss],
                    "plain": plain["bound"], "clipped": clipped["bound"],
                    "selected_clip_ratio": clipped["selected_clip"] / term_cap,
                })

    def summarize(selected):
        truth = np.asarray([row["truth"] for row in selected])
        result = {}
        for method in ("plain", "clipped"):
            bounds = np.asarray([row[method] for row in selected])
            valid = bounds <= truth + 1e-12
            result[method] = {
                "valid": int(valid.sum()), "total": int(valid.size),
                "wilson_lower_95": _wilson_lower(int(valid.sum()), int(valid.size)),
                "median_usable_tightness": float(np.median(
                    np.where(valid, bounds / truth, 0.0)
                )),
            }
        clipped = np.asarray([row["clipped"] for row in selected])
        plain = np.asarray([row["plain"] for row in selected])
        result["paired_gain"] = {
            "strict_clipped_wins": int(np.sum(clipped > plain + 1e-12)),
            "strict_plain_wins": int(np.sum(plain > clipped + 1e-12)),
            "median_clipped_minus_plain": float(np.median(clipped - plain)),
        }
        return result

    return {
        "status": "exploratory_known_truth_comparison",
        "pythia_trained_models": False,
        "seed": seed, "n_entities": n_entities,
        "repetitions_per_design": repetitions,
        "clip_ratios_of_declared_term_range": [.125, .25, .5, 1.0],
        "clip_selection": "Bonferroni; grid fixed before each synthetic outcome draw",
        "summary": summarize(rows),
        "by_loss": {
            loss: summarize([row for row in rows if row["loss"] == loss])
            for loss in ("mae", "mse")
        },
        "by_design": {
            name: summarize([row for row in rows if row["design"] == name])
            for name in DESIGN_NAMES
        },
        "selected_clip_ratios": {
            str(ratio): int(sum(np.isclose(row["selected_clip_ratio"], ratio) for row in rows))
            for ratio in (.125, .25, .5, 1.0)
        },
    }
