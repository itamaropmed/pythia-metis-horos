"""Known-floor checks for direct repeated-environment SAA certificates."""
import json

import numpy as np

from repeated_environment_saa import (
    LOSSES,
    environment_coordinate_saa_portfolio,
    environment_dependence_aware_portfolio,
    environment_holdout_upper,
    environment_mse_unbiased_scores,
    environment_saa_portfolio,
)


def gold(probabilities, loss):
    p = np.asarray(probabilities)
    if loss == "MSE":
        return float(np.mean(p * (1 - p)))
    if loss == "MAE" or loss == "Q50":
        return float(np.mean(np.minimum(p, 1 - p) * (.5 if loss == "Q50" else 1.)))
    q = float(loss[1:]) / 100
    return float(np.mean(np.minimum(q * p, (1 - q) * (1 - p))))


def run(reps=100):
    violations = 0
    cases = 0
    positive = 0
    relative_errors = []
    selected = {}
    coordinate_violations = 0
    coordinate_strict_improvements = 0
    dependence_aware_violations = 0
    mse_unbiased_errors = []
    upper_violations = 0
    interval_contradictions = 0
    caps = {"MAE": .5, "MSE": .25, "Q10": .09, "Q50": .25, "Q90": .09}
    for seed in range(reps):
        rng = np.random.default_rng(seed + 663_001)
        p = rng.uniform(.05, .95, 40)
        environments = (rng.random((2_000, len(p))) < p[None, :]).astype(float)
        for loss in LOSSES:
            result = environment_saa_portfolio(
                environments, loss=loss, score_cap=caps[loss],
                ks=(2, 4, 8, 16, 32), clip=(0., 1.),
                alpha_family=.05 / len(LOSSES),
                environment_provenance="known Bernoulli repeated environments")
            coordinate = environment_coordinate_saa_portfolio(
                environments, loss=loss, score_cap=caps[loss],
                ks=(2, 4, 8, 16, 32), clip=(0., 1.),
                alpha_family=.05 / len(LOSSES),
                environment_provenance="known Bernoulli repeated environments",
                coordinate_independence_provenance="independent Bernoulli RNG draws")
            dependence_aware = environment_dependence_aware_portfolio(
                environments, loss=loss, score_cap=caps[loss],
                ks=(2, 4, 8, 16, 32), clip=(0., 1.),
                alpha_family=.05 / len(LOSSES),
                environment_provenance="known Bernoulli repeated environments",
                coordinate_independence_provenance="independent Bernoulli RNG draws")
            upper = environment_holdout_upper(
                environments, loss=loss, clip=(0., 1.),
                alpha_family=.05 / len(LOSSES),
                environment_provenance="known Bernoulli repeated environments")
            truth = gold(p, loss)
            violations += result["bound"] > truth + 1e-12
            coordinate_violations += coordinate["bound"] > truth + 1e-12
            dependence_aware_violations += dependence_aware["bound"] > truth + 1e-12
            coordinate_strict_improvements += coordinate["bound"] > result["bound"]
            upper_violations += upper["upper"] < truth - 1e-12
            interval_contradictions += result["bound"] > upper["upper"] + 1e-12
            positive += result["bound"] > 0
            relative_errors.append(abs(result["plugin_clipped"] - truth) / truth)
            key = (result["selected_k"], result["selected_method"])
            selected[key] = selected.get(key, 0) + 1
            if loss == "MSE":
                unbiased = environment_mse_unbiased_scores(
                    environments, k=8, clip=(0., 1.))
                mse_unbiased_errors.append(abs(np.mean(unbiased) - truth) / truth)
                assert result["selected_witness"] == "unbiased_variance"
            cases += 1
    refused = environment_saa_portfolio(
        environments, loss="MAE", score_cap=.5, clip=(0., 1.),
        environment_provenance=None)
    assert refused["status"] == "refused"
    refused_coordinate = environment_coordinate_saa_portfolio(
        environments, loss="MAE", score_cap=.5, clip=(0., 1.),
        environment_provenance="known repeated environments",
        coordinate_independence_provenance=None)
    assert refused_coordinate["status"] == "refused"
    window_violations = 0
    window_positive = 0
    window_cases = 0
    for seed in range(20):
        rng = np.random.default_rng(seed + 664_001)
        environments = 100 * rng.random((2_000, 40))
        for loss, truth in (("W15", .7), ("W30", .4), ("W60", 0.)):
            result = environment_saa_portfolio(
                environments, loss=loss, score_cap=1., ks=(2, 4, 8, 16),
                clip=(0., 100.), alpha_family=.05 / 3,
                environment_provenance="known uniform repeated environments")
            window_violations += result["bound"] > truth + 1e-12
            window_positive += result["bound"] > 0
            window_cases += 1
    # Directly compare the vectorized window block implementation against the
    # known zero floor when a +/-60 window covers the entire [0,100] clip.
    assert (violations == 0 and coordinate_violations == 0
            and dependence_aware_violations == 0 and upper_violations == 0
            and interval_contradictions == 0 and window_violations == 0)
    out = {"repetitions": reps, "loss_cases": cases,
           "known_gold_exceedances": violations,
           "coordinate_route_known_gold_exceedances": coordinate_violations,
           "coordinate_route_strict_improvements": coordinate_strict_improvements,
           "dependence_aware_known_gold_exceedances": dependence_aware_violations,
           "upper_below_known_gold": upper_violations,
           "interval_contradictions": interval_contradictions,
           "positive_bounds": positive,
           "plugin_median_relative_error": float(np.median(relative_errors)),
           "mse_unbiased_score_median_relative_error": float(
               np.median(mse_unbiased_errors)),
           "selected_routes": {f"k{k}_{method}": n
                               for (k, method), n in selected.items()},
           "window_uniform_cases": window_cases,
           "window_known_gold_exceedances": window_violations,
           "window_positive_bounds": window_positive,
           "missing_provenance_refused": True}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
