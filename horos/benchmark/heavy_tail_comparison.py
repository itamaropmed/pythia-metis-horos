"""Finite-variance heavy-tail comparison for patient-level Pythia bounds."""
from __future__ import annotations

import numpy as np

from pythia.core.bregman import bregman_bound
from pythia.core.dependence_bounds import (
    catoni_cluster_pairwise_bound,
    median_of_means_cluster_pairwise_bound,
)

from .dependence_comparison import _wilson_lower


def run_heavy_tail_comparison(*, seed: int = 20260913, repetitions: int = 300) -> dict:
    rows = []
    sample_sizes = (500, 2_000, 10_000)
    noise_scale, degrees = 0.3, 5
    common_scale = 0.15
    idiosyncratic = (0.1, 0.2, 0.3)
    r_star = noise_scale ** 2 * degrees / (degrees - 2)
    noise_fourth = noise_scale ** 4 * 3 * degrees ** 2 / ((degrees - 2) * (degrees - 4))
    common_variance = common_scale ** 2
    error_variances = np.asarray([common_variance + value ** 2 for value in idiosyncratic])
    risk_variances = (
        noise_fourth + 6 * r_star * error_variances + 3 * error_variances ** 2
        - (r_star + error_variances) ** 2
    )
    disagreement_variances = np.asarray([
        2 * (idiosyncratic[i] ** 2 + idiosyncratic[j] ** 2) ** 2
        for i in range(3) for j in range(i + 1, 3)
    ])
    lambda_cap = max(
        common_variance / np.sqrt(error_variances[i] * error_variances[j])
        for i in range(3) for j in range(i + 1, 3)
    )
    seeds = np.random.SeedSequence(seed).spawn(len(sample_sizes) * repetitions)
    cursor = 0
    for n in sample_sizes:
        for repetition in range(repetitions):
            rng = np.random.default_rng(seeds[cursor]); cursor += 1
            signal = rng.normal(size=n)
            noise = noise_scale * rng.standard_t(degrees, size=n)
            common = rng.normal(scale=common_scale, size=n)
            errors = np.vstack([
                common + rng.normal(scale=value, size=n) for value in idiosyncratic
            ])
            target = signal + noise
            predictions = signal[None, :] + errors
            kwargs = dict(
                y=target, predictions=predictions, groups=np.arange(n),
                lambda_max=lambda_cap,
                risk_variance_bound=float(risk_variances.max() * 1.001),
                disagreement_variance_bound=float(disagreement_variances.max() * 1.001),
            )
            catoni = catoni_cluster_pairwise_bound(**kwargs)
            mom = median_of_means_cluster_pairwise_bound(**kwargs, seed=repetition)
            legacy = bregman_bound(target, predictions, "mse")
            rows.append({"n": n, "catoni": catoni["bound"], "mom": mom["bound"],
                         "legacy": legacy["bound"]})

    def summarize(selected: list[dict]) -> dict:
        result = {}
        for method in ("catoni", "mom", "legacy"):
            values = np.asarray([row[method] for row in selected])
            valid = values <= r_star + 1e-12
            result[method] = {
                "valid": int(valid.sum()), "total": int(values.size),
                "coverage": float(valid.mean()),
                "wilson_lower_95": _wilson_lower(int(valid.sum()), int(values.size)),
                "median_usable_tightness": float(np.median(np.where(valid, values / r_star, 0.0))),
            }
        return result

    return {
        "status": "exploratory_validation",
        "claim": "finite-variance iid-patient simulation with oracle variance and correlation caps",
        "pythia_trained_models": False,
        "seed": seed, "repetitions_per_sample_size": repetitions,
        "known_bayes_risk": r_star, "lambda_cap": lambda_cap,
        "n_runs": len(rows), "summary": summarize(rows),
        "by_sample_size": {
            str(n): summarize([row for row in rows if row["n"] == n]) for n in sample_sizes
        },
    }
