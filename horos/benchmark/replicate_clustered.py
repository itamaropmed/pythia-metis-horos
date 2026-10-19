"""Known-risk stress test for dependence within replicate clusters."""
from __future__ import annotations

import numpy as np

from pythia.core.replicates import paired_replicate_lower_bound

from .dependence_comparison import _wilson_lower


def run_replicate_clustered(
    *, seed: int = 20260917, repetitions: int = 500,
    n_blocks: int = 200, pairs_per_block: int = 20,
) -> dict:
    """Compare false pair independence with valid independent-block inference."""
    rng = np.random.default_rng(seed)
    pair_bounds, block_bounds = [], []
    true_risk = (0.1 ** 2 + 1.9 ** 2) / 6.0
    n_pairs = n_blocks * pairs_per_block
    blocks = np.repeat(np.arange(n_blocks), pairs_per_block)
    left = np.arange(n_pairs)
    right = np.arange(n_pairs, 2 * n_pairs)
    distance = np.zeros(n_pairs)
    for _ in range(repetitions):
        scale = rng.choice((0.1, 1.9), size=n_blocks)
        pair_scale = scale[blocks]
        yl = rng.uniform(-pair_scale, pair_scale)
        yr = rng.uniform(-pair_scale, pair_scale)
        kwargs = dict(
            y_left=yl, y_right=yr, distances=distance,
            left_entities=left, right_entities=right, pair_blocks=blocks,
            loss="mse", lipschitz_bound=0.0, distance_bound=1.0,
            outcome_difference_bound=3.8,
        )
        pair_bounds.append(paired_replicate_lower_bound(
            **kwargs, concentration="conditional_empirical_bernstein",
        )["bound"])
        block_bounds.append(paired_replicate_lower_bound(
            **kwargs, concentration="conditional_block_empirical_bernstein",
        )["bound"])

    def summary(values):
        values = np.asarray(values)
        valid = values <= true_risk + 1e-12
        return {
            "valid": int(valid.sum()), "total": int(values.size),
            "coverage": float(valid.mean()),
            "wilson_lower_95": _wilson_lower(int(valid.sum()), int(values.size)),
            "median_tightness": float(np.median(np.where(valid, values / true_risk, 0))),
        }

    return {
        "status": "exploratory_validation", "pythia_trained_models": False,
        "seed": seed, "n_blocks": n_blocks, "pairs_per_block": pairs_per_block,
        "repetitions": repetitions, "true_mse_bayes_risk": true_risk,
        "pair_independence_negative_control": summary(pair_bounds),
        "independent_block_method": summary(block_bounds),
    }
