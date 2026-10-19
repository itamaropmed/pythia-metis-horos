"""Model-free near-replicate simulations for MAE and MSE floor bounds."""
from __future__ import annotations

import numpy as np

from pythia.core.replicates import paired_replicate_lower_bound

from .dependence_comparison import _wilson_lower


def _simulate_cell(rng, n_blocks, slope, loss, noise_family,
                   noise_bound, distance_cap, true_risks) -> dict:
    x = rng.uniform(size=(n_blocks, 8))
    if noise_family == "uniform":
        noise = rng.uniform(-noise_bound, noise_bound, size=x.shape)
    else:
        noise = rng.choice(
            (-noise_bound, 0.0, noise_bound), size=x.shape,
            p=(.05, .90, .05),
        )
    y = slope * x + noise
    order = np.argsort(x, axis=1)
    x_sorted = np.take_along_axis(x, order, axis=1)
    y_sorted = np.take_along_axis(y, order, axis=1)
    pair_distances = x_sorted[:, 1::2] - x_sorted[:, 0::2]
    eligible = pair_distances <= distance_cap
    retained_blocks = np.any(eligible, axis=1)
    eligible = eligible[retained_blocks]
    pair_distances = pair_distances[retained_blocks]
    left_values = y_sorted[retained_blocks, 0::2]
    right_values = y_sorted[retained_blocks, 1::2]
    yl, yr = left_values[eligible], right_values[eligible]
    distances = pair_distances[eligible]
    n_pairs = yl.size
    left = np.arange(2 * n_pairs)[0::2]
    right = np.arange(2 * n_pairs)[1::2]
    block_grid = np.broadcast_to(
        np.arange(int(retained_blocks.sum()))[:, None], eligible.shape
    )
    blocks = block_grid[eligible]
    kwargs = dict(
        y_left=yl, y_right=yr, distances=distances,
        left_entities=left, right_entities=right, pair_blocks=blocks,
        loss=loss, distance_bound=distance_cap,
        outcome_difference_bound=slope * distance_cap + 2 * noise_bound,
    )
    corrected = paired_replicate_lower_bound(**kwargs, lipschitz_bound=slope)
    hoeffding = paired_replicate_lower_bound(
        **kwargs, lipschitz_bound=slope, concentration="hoeffding",
    )
    half_lipschitz = paired_replicate_lower_bound(
        **kwargs, lipschitz_bound=.5 * slope,
    )
    naive = paired_replicate_lower_bound(**kwargs, lipschitz_bound=0.0)
    return {
        "n_blocks": n_blocks, "n_entities": 8 * n_blocks,
        "n_retained_blocks": int(retained_blocks.sum()), "n_pairs": int(n_pairs),
        "slope": slope, "loss": loss, "noise_family": noise_family,
        "true_risk": true_risks[noise_family][loss],
        "corrected": corrected["bound"], "corrected_estimate": corrected["estimate"],
        "hoeffding": hoeffding["bound"],
        "half_lipschitz": half_lipschitz["bound"], "naive": naive["bound"],
    }


def run_replicate_comparison(*, seed: int = 20260915, repetitions: int = 100) -> dict:
    """Test Lipschitz-corrected replicate bounds against naive pairing."""
    block_counts = (125, 500, 1_250)  # 1k, 4k, 10k entities at block size eight
    slopes = (0.0, 1.0, 3.0)
    losses = ("mae", "mse")
    noise_families = ("uniform", "spike_mixture")
    noise_bound = 0.7
    distance_cap = 0.25
    true_risks = {
        "uniform": {"mae": noise_bound / 2.0, "mse": noise_bound ** 2 / 3.0},
        "spike_mixture": {"mae": .1 * noise_bound, "mse": .1 * noise_bound ** 2},
    }
    seeds = np.random.SeedSequence(seed).spawn(
        len(block_counts) * len(slopes) * len(losses) * len(noise_families) * repetitions
    )
    rows = []
    cursor = 0
    for n_blocks in block_counts:
        for slope in slopes:
            for loss in losses:
                for noise_family in noise_families:
                    for repetition in range(repetitions):
                        rng = np.random.default_rng(seeds[cursor]); cursor += 1
                        rows.append(_simulate_cell(
                            rng, n_blocks, slope, loss, noise_family,
                            noise_bound, distance_cap, true_risks,
                        ))

    def summarize(selected: list[dict]) -> dict:
        result = {}
        for method in ("corrected", "hoeffding", "half_lipschitz", "naive"):
            values = np.asarray([row[method] for row in selected])
            truths = np.asarray([row["true_risk"] for row in selected])
            valid = values <= truths + 1e-12
            result[method] = {
                "valid": int(valid.sum()), "total": int(values.size),
                "coverage": float(valid.mean()),
                "wilson_lower_95": _wilson_lower(int(valid.sum()), int(values.size)),
                "median_usable_tightness": float(np.median(np.where(valid, values / truths, 0.0))),
            }
        corrected = np.asarray([row["corrected"] for row in selected])
        hoeffding = np.asarray([row["hoeffding"] for row in selected])
        result["empirical_bernstein_gain"] = {
            "strict_wins": int(np.sum(corrected > hoeffding + 1e-12)),
            "total": int(corrected.size),
            "median_gain": float(np.median(corrected - hoeffding)),
        }
        estimates = np.asarray([row["corrected_estimate"] for row in selected])
        truths = np.asarray([row["true_risk"] for row in selected])
        result["corrected_estimate"] = {
            "median_relative_error": float(np.median(np.abs(estimates - truths) / truths)),
            "median_signed_error": float(np.median(estimates - truths)),
        }
        return result

    return {
        "status": "exploratory_validation",
        "claim": "iid randomized-block near-replicate simulation with known Lipschitz and range caps",
        "pythia_trained_models": False, "seed": seed,
        "repetitions_per_cell": repetitions, "n_runs": len(rows),
        "summary": summarize(rows),
        "by_loss": {
            loss: summarize([row for row in rows if row["loss"] == loss])
            for loss in losses
        },
        "by_entity_count": {
            str(8 * count): summarize([row for row in rows if row["n_blocks"] == count])
            for count in block_counts
        },
        "by_slope": {
            str(slope): summarize([row for row in rows if row["slope"] == slope])
            for slope in slopes
        },
        "by_noise_family": {
            family: summarize([row for row in rows if row["noise_family"] == family])
            for family in noise_families
        },
    }
