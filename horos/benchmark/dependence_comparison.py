"""Held-out bounded simulations comparing Pythia dependence rungs."""
from __future__ import annotations

import numpy as np

from pythia.core.bregman import bregman_bound
from pythia.core.dependence_bounds import (
    bounded_sample_pairwise_bound,
    sharp_pairwise_population_bound,
)


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((center - radius) / denominator)


def run_dependence_comparison(
    *, seed: int = 20260912, repetitions: int = 200,
) -> dict:
    """Compare validity and efficiency without fitting any predictive model."""
    rows = []
    sample_sizes = (500, 2_000, 10_000)
    common_bounds = (0.0, 0.2, 0.5)
    noise_bound = 0.7
    idiosyncratic_bounds = (0.15, 0.25, 0.35)
    r_star = noise_bound ** 2 / 3.0
    sequence = np.random.SeedSequence(seed).spawn(len(sample_sizes) * len(common_bounds) * repetitions)
    cursor = 0
    for n in sample_sizes:
        for common_bound in common_bounds:
            common_variance = common_bound ** 2 / 3.0
            error_variances = np.asarray([
                common_variance + value ** 2 / 3.0 for value in idiosyncratic_bounds
            ])
            lambda_cap = max(
                common_variance / np.sqrt(error_variances[i] * error_variances[j])
                if common_variance else 0.0
                for i in range(3) for j in range(i + 1, 3)
            )
            absolute_cap = noise_bound + common_bound + max(idiosyncratic_bounds)
            for repetition in range(repetitions):
                rng = np.random.default_rng(sequence[cursor]); cursor += 1
                signal = rng.uniform(-1.0, 1.0, n)
                noise = rng.uniform(-noise_bound, noise_bound, n)
                common = rng.uniform(-common_bound, common_bound, n)
                errors = np.vstack([
                    common + rng.uniform(-value, value, n)
                    for value in idiosyncratic_bounds
                ])
                predictions = signal[None, :] + errors
                target = signal + noise
                legacy = bregman_bound(target, predictions, "mse")
                empirical_risks = np.mean((predictions - target[None, :]) ** 2, axis=1)
                disagreements = np.mean(
                    (predictions[:, None, :] - predictions[None, :, :]) ** 2, axis=2
                )
                plugin = sharp_pairwise_population_bound(
                    empirical_risks, disagreements, lambda_cap
                )
                eb = bounded_sample_pairwise_bound(
                    target, predictions, lambda_cap, absolute_cap,
                    alpha=0.05, concentration="empirical_bernstein",
                )
                hoeffding = bounded_sample_pairwise_bound(
                    target, predictions, lambda_cap, absolute_cap,
                    alpha=0.05, concentration="hoeffding",
                )
                rows.append({
                    "n": n, "common_bound": common_bound, "repetition": repetition,
                    "r_star": r_star, "lambda_cap": lambda_cap,
                    "legacy_bound": float(legacy["bound"]),
                    "legacy_estimator": float(legacy["estimator"]),
                    "sharp_plugin_estimator": plugin["bound"],
                    "empirical_bernstein_bound": eb["bound"],
                    "hoeffding_bound": hoeffding["bound"],
                })

    def summarize(selected: list[dict]) -> dict:
        methods = (
            "legacy_bound", "empirical_bernstein_bound", "hoeffding_bound",
        )
        result = {}
        for method in methods:
            values = np.asarray([row[method] for row in selected], float)
            valid = values <= r_star + 1e-12
            result[method] = {
                "valid": int(valid.sum()), "total": int(values.size),
                "coverage": float(valid.mean()),
                "wilson_lower_95": _wilson_lower(int(valid.sum()), int(values.size)),
                "median_tightness": float(np.median(np.where(valid, values / r_star, 0.0))),
            }
        for method in ("legacy_estimator", "sharp_plugin_estimator"):
            values = np.asarray([row[method] for row in selected], float)
            result[method] = {
                "median_absolute_error": float(np.median(np.abs(values - r_star))),
                "median_relative_error": float(np.median(np.abs(values - r_star) / r_star)),
                "median_signed_error": float(np.median(values - r_star)),
            }
        eb = np.asarray([row["empirical_bernstein_bound"] for row in selected])
        hf = np.asarray([row["hoeffding_bound"] for row in selected])
        result["empirical_bernstein_tighter_than_hoeffding"] = {
            "count": int(np.sum(eb > hf + 1e-12)), "total": int(eb.size),
            "median_gain": float(np.median(eb - hf)),
        }
        return result

    by_n = {
        str(n): summarize([row for row in rows if row["n"] == n])
        for n in sample_sizes
    }
    return {
        "status": "exploratory_validation",
        "claim": "bounded-iid simulation; finite-sample rung is conditional on declared assumptions",
        "pythia_trained_models": False,
        "seed": seed, "repetitions_per_cell": repetitions,
        "n_cells": len(sample_sizes) * len(common_bounds),
        "n_runs": len(rows), "known_bayes_risk": r_star,
        "summary": summarize(rows), "by_sample_size": by_n,
    }
