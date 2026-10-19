"""Known-law checks for repeated-environment transport caps."""
import json

import numpy as np
from scipy.stats import norm

from coupled_environment_w1 import (
    calibrate_coupled_mean_w1,
    environment_costs,
)


def run(reps=100):
    rng_design = np.random.default_rng(884_901)
    n, k = 48, 4
    blocks = np.arange(n).reshape(-1, k)
    mu = rng_design.uniform(-8, 8, n)
    scale = rng_design.uniform(2, 7, n)
    clip = (-30., 30.)
    u_grid = (np.arange(100_000) + .5) / 100_000
    exact_replicates = mu[None, :] + scale[None, :] * norm.ppf(u_grid)[:, None]
    true_mean_w1 = float(np.mean(environment_costs(
        exact_replicates, blocks, clip=clip)[0]))

    common_misses = independent_misses = 0
    common_caps, independent_caps = [], []
    methods = {}
    for seed in range(reps):
        rng = np.random.default_rng(seed + 884_902)
        environments = 1_000
        common_z = norm.ppf(rng.random(environments))
        common = mu[None, :] + scale[None, :] * common_z[:, None]
        independent = mu[None, :] + scale[None, :] * norm.ppf(
            rng.random((environments, n)))
        common_result = calibrate_coupled_mean_w1(
            common, blocks, clip=clip, delta_family=.01,
            coupling_provenance="registered common-quantile synthetic environments")
        independent_result = calibrate_coupled_mean_w1(
            independent, blocks, clip=clip, delta_family=.01,
            coupling_provenance="registered independent synthetic environments")
        common_caps.append(common_result["mean_w1_cap"])
        independent_caps.append(independent_result["mean_w1_cap"])
        common_misses += common_result["mean_w1_cap"] < true_mean_w1 - 2e-4
        independent_misses += independent_result["mean_w1_cap"] < true_mean_w1 - 2e-4
        methods[common_result["selected_method"]] = methods.get(
            common_result["selected_method"], 0) + 1
    refusal = calibrate_coupled_mean_w1(
        common, blocks, clip=clip, coupling_provenance=None)
    assert refusal["status"] == "refused"
    assert common_misses == independent_misses == 0
    assert np.median(common_caps) < np.median(independent_caps)
    out = {
        "repetitions": reps, "environments_per_repetition": 1_000,
        "true_mean_clipped_w1": true_mean_w1,
        "common_coupling_cap_misses": common_misses,
        "independent_coupling_cap_misses": independent_misses,
        "median_common_cap": float(np.median(common_caps)),
        "median_independent_cap": float(np.median(independent_caps)),
        "common_selected_methods": methods,
        "missing_provenance_refused": True,
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
