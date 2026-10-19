"""Reference and known-mean checks for the external dKelly route."""
import json

import numpy as np

from repeated_environment_saa import dkelly_betting_lower_mean


def brute_grid(values, alpha, d=10, breaks=20_000):
    """Literal final-time grid inversion following confseq's dKelly code."""
    x = np.asarray(values, float)
    c = np.arange(1, d + 1) / (d + 1)
    grid = np.arange(1, breaks + 1) / breaks
    accepted = []
    for m in grid:
        capital = np.mean(np.prod(
            (1 - c[:, None]) + c[:, None] * x[None, :] / m, axis=1))
        if capital <= 1 / alpha:
            accepted.append(m)
            break
    return max(0., (accepted[0] if accepted else 1 / breaks) - 1 / breaks)


def run(repetitions=2_000):
    rng = np.random.default_rng(771_001)
    violations = 0
    gaps = []
    for _ in range(repetitions):
        probabilities = rng.uniform(.02, .98, 80)
        values = (rng.random(80) < probabilities).astype(float)
        bound = dkelly_betting_lower_mean(values, 1., .05)
        violations += bound > np.mean(probabilities) + 1e-12
    for seed in range(10):
        local = np.random.default_rng(772_001 + seed)
        values = local.beta(2, 5, 30)
        exact = dkelly_betting_lower_mean(values, 1., .05)
        grid = brute_grid(values, .05)
        gaps.append(abs(exact - grid))
    deterministic = dkelly_betting_lower_mean(np.full(100, .3), 1., .05)
    assert violations <= 140  # loose implementation smoke gate, not the proof
    assert max(gaps) <= 1.1 / 20_000
    assert 0 < deterministic < .3
    out = {
        "heterogeneous_bernoulli_repetitions": repetitions,
        "known_average_mean_exceedances": int(violations),
        "empirical_exceedance_rate": float(violations / repetitions),
        "maximum_reference_grid_gap": float(max(gaps)),
        "deterministic_point_three_lower": float(deterministic),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
