"""Population-style stress tests for the exploratory squared-loss profile."""
from __future__ import annotations

import numpy as np

from pythia.core.dependence import dependence_profile


def run_dependence_suite(*, seed: int = 20260910, n: int = 12000) -> dict:
    """Vary common bias independently of outcome noise, SNR, and error family.

    The supplied cap is calculated from latent approximation errors available in
    simulation only. A violation means the returned lower bound exceeds the
    known Bayes risk. Passing this suite is a regression check, not a proof.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for signal_scale in (0.5, 2.0):
        for noise_scale in (0.2, 0.7):
            for common_scale in (0.0, 0.25, 0.75):
                for family in ("gaussian", "laplace"):
                    signal = signal_scale * rng.normal(size=n)
                    outcome_noise = noise_scale * rng.normal(size=n)
                    common = common_scale * rng.normal(size=n)
                    draws = (
                        [rng.normal(scale=s, size=n) for s in (0.15, 0.25, 0.4)]
                        if family == "gaussian" else
                        [rng.laplace(scale=s / np.sqrt(2), size=n) for s in (0.15, 0.25, 0.4)]
                    )
                    approximation = np.vstack([common + value for value in draws])
                    predictions = signal[None, :] + approximation
                    target = signal + outcome_noise
                    corr = np.corrcoef(approximation)
                    cap = float(max(corr[i, j] for i in range(3) for j in range(i + 1, 3)))
                    cap = min(max(cap, 0.0) + 1e-8, 1.0 - 1e-8)
                    profile = dependence_profile(target, predictions, np.array([cap]))
                    bound = float(profile["curve"][0]["bound"])
                    oracle = float(np.mean(outcome_noise ** 2))
                    rows.append({
                        "signal_scale": signal_scale, "noise_scale": noise_scale,
                        "common_scale": common_scale, "family": family,
                        "lambda_cap": cap, "bound": bound,
                        "oracle_bayes_risk": oracle,
                        "violation": bool(bound > oracle + 1e-6),
                    })
    violations = sum(row["violation"] for row in rows)
    return {
        "status": "exploratory_validation",
        "seed": seed, "n_per_design": n, "n_designs": len(rows),
        "violations": violations, "passed": violations == 0, "designs": rows,
        "claim": "simulation regression check; not a theorem or finite-sample guarantee",
    }
