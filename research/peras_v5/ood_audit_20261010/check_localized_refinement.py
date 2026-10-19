"""Known-noise partial-disclosure checks for localized sufficiency curves."""
import json

import numpy as np

from finite_group_multiloss_interval import LOSSES, SCORE_CAPS, empirical_floor_all, interval
from refinement_gap_profile import localized_refinement_profile, maximum_clipped_bayes_risk


def run(reps=200):
    probabilities = np.array([.5, .51])
    group_gold = {"MAE": 80 * np.minimum(probabilities, 1 - probabilities),
                  "MSE": 6400 * probabilities * (1 - probabilities),
                  "Q10": 8 * probabilities,
                  "Q50": 40 * np.minimum(probabilities, 1 - probabilities),
                  "Q90": 8 * (1 - probabilities),
                  "W15": np.minimum(probabilities, 1 - probabilities),
                  "W30": np.minimum(probabilities, 1 - probabilities),
                  "W60": np.zeros(2)}
    violations = 0
    false_zero_positive = 0
    cases = 0
    rng_max = np.random.default_rng(991_007)
    for _ in range(1_000):
        y = rng_max.uniform(-40, 40, rng_max.integers(2, 200))
        for loss in LOSSES:
            assert empirical_floor_all(y, loss) <= maximum_clipped_bayes_risk(
                loss, (-40, 40)) + 1e-10
    for seed in range(reps):
        rng = np.random.default_rng(seed + 830_000)
        calibration = {g: rng.choice((-40., 40.), 4_000, p=(1 - p, p))
                       for g, p in enumerate(probabilities)}
        groups = np.tile((0, 1), 200)
        weights = {0: .5, 1: .5}
        results = {loss: interval(calibration, weights, loss=loss,
                                  clip=(-40, 40), score_cap=SCORE_CAPS[loss])
                   for loss in LOSSES}
        for nominal in (0., .25, .5, .75, 1.):
            disclosed = rng.random(len(groups)) < nominal
            epsilon = float(disclosed.mean())
            for loss in LOSSES:
                full_gold = float(np.mean((~disclosed) * group_gold[loss][groups]))
                profile = localized_refinement_profile(results[loss], [epsilon])
                bound = profile["bounds"][0]
                violations += bound > full_gold + 1e-12
                cases += 1
                if nominal == 1:
                    assert full_gold == 0 and bound == 0
                    false = localized_refinement_profile(results[loss], [0.])["bounds"][0]
                    false_zero_positive += false > 0
    assert violations == 0
    out = {"reps": reps, "loss_cases": cases,
           "maximum_risk_random_checks": 1_000 * len(LOSSES),
           "localized_bound_gold_exceedances": violations,
           "false_epsilon_zero_positive_at_full_disclosure": false_zero_positive}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
