"""Known-floor checks for count-routed partial fine-group certificates."""
import json

import numpy as np

from finite_group_multiloss_interval import (
    LOSSES,
    SCORE_CAPS,
    partial_group_lower,
    partial_group_portfolio,
)


def endpoint_gold(p, loss):
    # Y in {-40,40}; actions are unrestricted but optimizers lie in the hull.
    if loss == "MAE":
        return 80 * min(p, 1 - p)
    if loss == "MSE":
        return 6400 * p * (1 - p)
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        return min(q * 80 * p, (1 - q) * 80 * (1 - p))
    if loss.startswith("W"):
        h = float(loss[1:])
        return 0. if h >= 40 else min(p, 1 - p)
    raise ValueError(loss)


def run(reps=500):
    probabilities = {0: .12, 1: .35, 2: .52, 3: .81, 4: .95}
    weights = {0: .15, 1: .2, 2: .25, 3: .25, 4: .15}
    violations = 0
    cases = 0
    positive = 0
    portfolio_violations = 0
    portfolio_positive = 0
    portfolio_tighter = 0
    omitted = []
    for seed in range(reps):
        rng = np.random.default_rng(seed + 776_001)
        sizes = {0: 128, 1: 64, 2: 31, 3: 15, 4: 0}
        samples = {g: rng.choice((-40., 40.), sizes[g],
                                 p=(1 - probabilities[g], probabilities[g]))
                   for g in sizes if sizes[g]}
        for loss in LOSSES:
            result = partial_group_lower(
                samples, weights, loss=loss, clip=(-40., 40.), k=16,
                score_cap=min(SCORE_CAPS[loss], 6400.),
                alpha_endpoint=.05 / len(LOSSES))
            portfolio = partial_group_portfolio(
                samples, weights, loss=loss, ks=(2, 4, 8, 16),
                clip=(-40., 40.), score_cap=min(SCORE_CAPS[loss], 6400.),
                alpha_family=.05 / len(LOSSES))
            gold = sum(weights[g] * endpoint_gold(probabilities[g], loss)
                       for g in weights)
            violations += result["lower"] > gold + 1e-12
            portfolio_violations += portfolio["lower"] > gold + 1e-12
            positive += result["lower"] > 0
            portfolio_positive += portfolio["lower"] > 0
            portfolio_tighter += portfolio["lower"] > result["lower"] + 1e-12
            omitted.append(result["omitted_target_mass"])
            assert result["included_groups"] == 3  # 31 contributes one block.
            assert abs(result["omitted_target_mass"] - .40) < 1e-12
            cases += 1
    assert violations == 0 and portfolio_violations == 0
    out = {"reps": reps, "loss_cases": cases, "known_gold_exceedances": violations,
           "positive_bounds": positive,
           "portfolio_known_gold_exceedances": portfolio_violations,
           "portfolio_positive_bounds": portfolio_positive,
           "portfolio_strictly_tighter_than_k16": portfolio_tighter,
           "omitted_target_mass": float(np.median(omitted))}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
