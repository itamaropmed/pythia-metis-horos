"""Frozen battery registered in ACTION_DISPERSION_CONFIRMATION_REGISTRATION_20261014.md."""
from __future__ import annotations

import json
import sys

import numpy as np

from peras import bayes_action_jensen_dispersion, calibrate_repeated_action_dispersion


def _geometry(name, x):
    if name == "linear":
        return 1.5 * x, np.repeat(1.0, len(x))
    if name == "nonlinear":
        return np.sin(np.pi * x) + .4 * x * x, np.repeat(.8, len(x))
    if name == "sparse_step":
        return np.where(x > .55, 1.2, np.where(x < -.55, -1.2, 0)), np.repeat(.7, len(x))
    if name == "high_frequency":
        return np.sin(4 * np.pi * x), np.repeat(.65, len(x))
    if name == "heteroscedastic":
        return x, .25 + 1.2 * np.abs(x)
    if name == "discrete_cell":
        return np.round(2 * x) / 2, np.repeat(.75, len(x))
    if name == "interaction":
        return 1.3 * x * np.roll(x, 1), .5 + .4 * (x > 0)
    if name == "skewed_action":
        return np.exp(x) - 1.2, .4 + .3 * (x + 1)
    if name == "high_snr":
        return 2 * x, np.repeat(.15, len(x))
    if name == "low_snr":
        return .3 * x, np.repeat(1.8, len(x))
    raise ValueError(name)


def _truth(actions, kappas, blocks):
    variances = []
    mads = []
    maxima = []
    curvatures = []
    for block in blocks:
        a = actions[block]
        variances.append(float(np.var(a)))
        mads.append(float(np.mean(np.abs(a - np.median(a)))))
        maxima.append(float(np.max(np.abs(a - a.mean()))))
        curvatures.append(
            bayes_action_jensen_dispersion(a, kappas[block]).quadratic_penalty)
    return tuple(float(np.mean(v)) for v in (variances, mads, maxima, curvatures))


def _one(name, seed, q):
    rng = np.random.default_rng(seed)
    x = np.linspace(-.95, .95, 16)
    location, scale = _geometry(name, x)
    # The average of two U[-1,1] shocks has an analytic triangular law.
    zq = (np.sqrt(2 * q) - 1 if q <= .5
          else 1 - np.sqrt(2 * (1 - q)))
    actions = location + scale * zq
    shared = rng.uniform(-1, 1, size=(800, 1))
    independent = rng.uniform(-1, 1, size=(800, 16))
    outcomes = location + scale * .5 * (shared + independent)
    blocks = np.arange(16).reshape(2, 8)
    # Triangular marginal density has maximum 1 in z coordinates.  MAE has
    # twice the outcome density; pinball has the density itself.
    multiplier = 2.0 if q == .5 else 1.0
    kappas = multiplier / scale
    support = (float(np.min(location - scale)), float(np.max(location + scale)))
    result = calibrate_repeated_action_dispersion(
        outcomes, blocks, quantile=q, alpha=.01, support=support,
        environment_provenance="iid-row-shared-uniform-v1",
        support_provenance="analytic-location-scale-support-v1",
        curvature_bounds=kappas,
        curvature_provenance="analytic-triangular-density-envelope-v1",
        independence_attested=True,
    )
    truth = _truth(actions, kappas, blocks)
    caps = (result.action_variance_cap, result.action_mad_cap,
            result.max_action_deviation_cap, result.curvature_penalty_cap)
    labels = ("variance", "mad", "max_deviation", "curvature_penalty")
    return {
        "mechanism": name, "seed": seed, "quantile": q,
        "n_environments": 800, "alpha": .01,
        **{f"true_{label}": value for label, value in zip(labels, truth)},
        **{f"cap_{label}": value for label, value in zip(labels, caps)},
        **{f"miss_{label}": cap + 1e-12 < value
           for label, value, cap in zip(labels, truth, caps)},
    }


if __name__ == "__main__":
    output = sys.argv[1]
    mechanisms = (
        "linear", "nonlinear", "sparse_step", "high_frequency",
        "heteroscedastic", "discrete_cell", "interaction", "skewed_action",
        "high_snr", "low_snr",
    )
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism in mechanisms:
            for run_seed in range(51001, 51051):
                for q in (.5, .9):
                    handle.write(json.dumps(_one(mechanism, run_seed, q), sort_keys=True) + "\n")
            print(mechanism, flush=True)
