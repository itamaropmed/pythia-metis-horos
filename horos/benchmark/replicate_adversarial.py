"""Adversarial known-risk designs for the model-free replicate theorem."""
from __future__ import annotations

import numpy as np

from horos.represent.pairing import global_nearest_pairs, randomized_block_pairs
from pythia.core.replicates import paired_replicate_lower_bound

from .dependence_comparison import _wilson_lower


DESIGN_NAMES = (
    "linear_uniform", "nonlinear_sine", "heteroskedastic",
    "asymmetric_mixture", "high_dimensional", "bounded_heavy_tail",
    "quadratic", "two_way_interaction", "sparse_ten_dimensional",
    "bimodal_noise",
)


def _design(rng: np.random.Generator, name: str, n: int):
    if name == "linear_uniform":
        x = rng.uniform(size=(n, 1)); center = 2.0 * x[:, 0]
        noise = rng.uniform(-.5, .5, n)
        return x, center + noise, 2.0, {"mae": .25, "mse": 1 / 12}, 3.0, {
            "mae": np.full(n, .25), "mse": np.full(n, 1 / 12),
        }
    if name == "nonlinear_sine":
        x = rng.uniform(size=(n, 1)); center = np.sin(2 * np.pi * x[:, 0])
        noise = rng.uniform(-.5, .5, n)
        return x, center + noise, 2 * np.pi, {"mae": .25, "mse": 1 / 12}, 3.0, {
            "mae": np.full(n, .25), "mse": np.full(n, 1 / 12),
        }
    if name == "heteroskedastic":
        x = rng.uniform(size=(n, 1)); scale = .2 + .6 * x[:, 0]
        noise = rng.uniform(-scale, scale)
        return x, x[:, 0] + noise, 1.0, {"mae": .25, "mse": .28 / 3}, 2.6, {
            "mae": scale / 2, "mse": scale ** 2 / 3,
        }
    if name == "asymmetric_mixture":
        x = rng.uniform(size=(n, 1)); center = 1.5 * x[:, 0]
        noise = rng.choice((-.2, .6), size=n, p=(.75, .25))
        return x, center + noise, 1.5, {"mae": .2, "mse": .12}, 2.3, {
            "mae": np.full(n, .2), "mse": np.full(n, .12),
        }
    if name == "high_dimensional":
        x = rng.uniform(size=(n, 5)); center = np.mean(np.sin(np.pi * x), axis=1)
        noise = rng.uniform(-.6, .6, n)
        return x, center + noise, np.pi, {"mae": .3, "mse": .12}, 2.2, {
            "mae": np.full(n, .3), "mse": np.full(n, .12),
        }
    if name == "bounded_heavy_tail":
        x = rng.uniform(size=(n, 1)); center = x[:, 0]
        noise = rng.choice((-2.0, 0.0, 2.0), size=n, p=(.02, .96, .02))
        return x, center + noise, 1.0, {"mae": .08, "mse": .16}, 5.0, {
            "mae": np.full(n, .08), "mse": np.full(n, .16),
        }
    if name == "quadratic":
        x = rng.uniform(size=(n, 1)); center = 2.0 * x[:, 0] ** 2
        noise = rng.uniform(-.5, .5, n)
        return x, center + noise, 4.0, {"mae": .25, "mse": 1 / 12}, 3.0, {
            "mae": np.full(n, .25), "mse": np.full(n, 1 / 12),
        }
    if name == "two_way_interaction":
        x = rng.uniform(size=(n, 2)); center = 2.0 * x[:, 0] * x[:, 1]
        noise = rng.uniform(-.5, .5, n)
        return x, center + noise, 4.0, {"mae": .25, "mse": 1 / 12}, 3.0, {
            "mae": np.full(n, .25), "mse": np.full(n, 1 / 12),
        }
    if name == "sparse_ten_dimensional":
        x = rng.uniform(size=(n, 10)); center = 3.0 * x[:, 0]
        noise = rng.uniform(-.5, .5, n)
        return x, center + noise, 30.0, {"mae": .25, "mse": 1 / 12}, 4.0, {
            "mae": np.full(n, .25), "mse": np.full(n, 1 / 12),
        }
    if name == "bimodal_noise":
        x = rng.uniform(size=(n, 1)); center = .5 * x[:, 0]
        noise = rng.choice((-.5, .5), size=n)
        return x, center + noise, .5, {"mae": .5, "mse": .25}, 1.5, {
            "mae": np.full(n, .5), "mse": np.full(n, .25),
        }
    raise ValueError(name)


def run_replicate_adversarial(*, seed: int = 20260916, repetitions: int = 50) -> dict:
    names = DESIGN_NAMES
    entity_counts = (1_000, 4_000, 10_000)
    losses = ("mae", "mse")
    seeds = np.random.SeedSequence(seed).spawn(
        len(names) * len(entity_counts) * len(losses) * repetitions
    )
    rows = []
    cursor = 0
    for name in names:
        for n in entity_counts:
            for loss in losses:
                for _ in range(repetitions):
                    rng = np.random.default_rng(seeds[cursor]); cursor += 1
                    x, y, lipschitz, risks, outcome_cap, conditional_risks = _design(rng, name, n)
                    entities = np.arange(n)
                    pairs = randomized_block_pairs(x, entities, block_size=8, seed=17)
                    kwargs = dict(
                        y_left=y[pairs.left_indices], y_right=y[pairs.right_indices],
                        distances=pairs.distances,
                        left_entities=entities[pairs.left_indices],
                        right_entities=entities[pairs.right_indices],
                        pair_blocks=pairs.pair_blocks, loss=loss,
                        distance_bound=1.0, outcome_difference_bound=outcome_cap,
                    )
                    corrected = paired_replicate_lower_bound(
                        **kwargs, lipschitz_bound=lipschitz,
                    )
                    hoeffding = paired_replicate_lower_bound(
                        **kwargs, lipschitz_bound=lipschitz, concentration="hoeffding",
                    )
                    conditional = paired_replicate_lower_bound(
                        **kwargs, lipschitz_bound=lipschitz,
                        concentration="conditional_hoeffding",
                    )
                    conditional_eb = paired_replicate_lower_bound(
                        **kwargs, lipschitz_bound=lipschitz,
                        concentration="conditional_empirical_bernstein",
                    )
                    global_pairs = global_nearest_pairs(x, entities)
                    global_conditional = paired_replicate_lower_bound(
                        y[global_pairs.left_indices], y[global_pairs.right_indices],
                        global_pairs.distances, entities[global_pairs.left_indices],
                        entities[global_pairs.right_indices], global_pairs.pair_blocks,
                        loss=loss, lipschitz_bound=lipschitz, distance_bound=1.0,
                        outcome_difference_bound=outcome_cap,
                        concentration="conditional_hoeffding",
                    )
                    global_conditional_eb = paired_replicate_lower_bound(
                        y[global_pairs.left_indices], y[global_pairs.right_indices],
                        global_pairs.distances, entities[global_pairs.left_indices],
                        entities[global_pairs.right_indices], global_pairs.pair_blocks,
                        loss=loss, lipschitz_bound=lipschitz, distance_bound=1.0,
                        outcome_difference_bound=outcome_cap,
                        concentration="conditional_empirical_bernstein",
                    )
                    half = paired_replicate_lower_bound(
                        **kwargs, lipschitz_bound=.5 * lipschitz,
                    )
                    naive = paired_replicate_lower_bound(**kwargs, lipschitz_bound=0.0)
                    rows.append({
                        "design": name, "n_entities": n, "loss": loss,
                        "true_risk": risks[loss], "corrected": corrected["bound"],
                        "conditional_true_risk": float(np.mean(conditional_risks[loss])),
                        "conditional_hoeffding": conditional["bound"],
                        "conditional_empirical_bernstein": conditional_eb["bound"],
                        "global_conditional": global_conditional["bound"],
                        "global_conditional_empirical_bernstein": global_conditional_eb["bound"],
                        "global_conditional_estimate": global_conditional_eb["estimate"],
                        "blocked_conditional_estimate": conditional_eb["estimate"],
                        "blocked_mean_distance": float(np.mean(pairs.distances)),
                        "global_mean_distance": float(np.mean(global_pairs.distances)),
                        "hoeffding": hoeffding["bound"], "half_lipschitz": half["bound"],
                        "naive": naive["bound"],
                    })

    def summarize(selected):
        result = {}
        for method in (
            "corrected", "hoeffding", "conditional_hoeffding",
            "conditional_empirical_bernstein", "global_conditional",
            "global_conditional_empirical_bernstein", "half_lipschitz", "naive",
        ):
            values = np.asarray([row[method] for row in selected])
            truth_key = (
                "conditional_true_risk"
                if method in {
                    "conditional_hoeffding", "conditional_empirical_bernstein",
                    "global_conditional", "global_conditional_empirical_bernstein",
                }
                else "true_risk"
            )
            truths = np.asarray([row[truth_key] for row in selected])
            valid = values <= truths + 1e-12
            result[method] = {
                "valid": int(valid.sum()), "total": int(values.size),
                "coverage": float(valid.mean()),
                "wilson_lower_95": _wilson_lower(int(valid.sum()), int(values.size)),
                "median_usable_tightness": float(np.median(np.where(valid, values / truths, 0))),
            }
        eb = np.asarray([row["corrected"] for row in selected])
        hf = np.asarray([row["hoeffding"] for row in selected])
        result["empirical_bernstein_gain"] = {
            "strict_wins": int(np.sum(eb > hf + 1e-12)), "total": int(eb.size),
            "median_gain": float(np.median(eb - hf)),
        }
        ch = np.asarray([row["conditional_hoeffding"] for row in selected])
        result["conditional_gain_over_population_eb"] = {
            "strict_wins": int(np.sum(ch > eb + 1e-12)), "total": int(eb.size),
            "median_gain": float(np.median(ch - eb)),
            "note": "different estimands: fixed audited covariates versus sampling population",
        }
        gc = np.asarray([row["global_conditional"] for row in selected])
        gceb = np.asarray([
            row["global_conditional_empirical_bernstein"] for row in selected
        ])
        ceb = np.asarray([row["conditional_empirical_bernstein"] for row in selected])
        result["global_pairing_gain"] = {
            "strict_wins_over_blocked_conditional": int(np.sum(gceb > ceb + 1e-12)),
            "total": int(gc.size), "median_bound_gain": float(np.median(gc - ch)),
            "median_empirical_bernstein_bound_gain": float(np.median(gceb - ceb)),
            "median_distance_ratio": float(np.median([
                row["global_mean_distance"] / row["blocked_mean_distance"]
                for row in selected
            ])),
        }
        result["conditional_empirical_bernstein_gain"] = {
            "strict_wins_over_conditional_hoeffding": int(np.sum(gceb > gc + 1e-12)),
            "total": int(gc.size), "median_gain": float(np.median(gceb - gc)),
        }
        conditional_truth = np.asarray([row["conditional_true_risk"] for row in selected])
        for method in ("global_conditional_estimate", "blocked_conditional_estimate"):
            estimates = np.asarray([row[method] for row in selected])
            result[method] = {
                "median_absolute_error": float(np.median(np.abs(estimates - conditional_truth))),
                "median_signed_error": float(np.median(estimates - conditional_truth)),
                "median_relative_error": float(np.median(
                    np.abs(estimates - conditional_truth) / conditional_truth
                )),
            }
        global_error = np.abs(np.asarray([
            row["global_conditional_estimate"] for row in selected
        ]) - conditional_truth)
        blocked_error = np.abs(np.asarray([
            row["blocked_conditional_estimate"] for row in selected
        ]) - conditional_truth)
        wins = int(np.sum(global_error + 1e-12 < blocked_error))
        result["global_estimator_vs_blocked"] = {
            "strict_wins": wins, "total": int(global_error.size),
            "wilson_lower_95_for_win_rate": _wilson_lower(wins, int(global_error.size)),
            "median_absolute_error_reduction": float(np.median(blocked_error - global_error)),
        }
        return result

    return {
        "status": "exploratory_validation", "pythia_trained_models": False,
        "seed": seed, "repetitions_per_cell": repetitions, "n_runs": len(rows),
        "summary": summarize(rows),
        "by_design": {name: summarize([row for row in rows if row["design"] == name]) for name in names},
        "by_loss": {loss: summarize([row for row in rows if row["loss"] == loss]) for loss in losses},
        "by_entity_count": {
            str(n): summarize([row for row in rows if row["n_entities"] == n])
            for n in entity_counts
        },
    }
