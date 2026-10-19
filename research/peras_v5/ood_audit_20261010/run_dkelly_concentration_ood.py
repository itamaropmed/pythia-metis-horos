"""Fresh external dKelly comparison on iid environment-block witnesses."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
import gold
from repeated_environment_saa import (
    METHODS,
    _loss_range,
    _lower_mean,
    environment_mse_unbiased_scores,
    environment_saa_scores,
)
from run_coupled_environment_ood import repeated_outcomes


LOSSES = ("MAE", "MSE", "Q10", "Q50", "Q90", "W15", "W30", "W60")
SCORE_CAPS = {
    "MAE": 30., "MSE": 1000., "Q10": 6., "Q50": 15., "Q90": 6.,
    "W15": 1., "W30": 1., "W60": 1.,
}
KS = (2, 4, 8, 16, 32)


def compare_methods(environments, loss):
    witnesses = (("empirical_optimum", "unbiased_variance")
                 if loss == "MSE" else ("empirical_optimum",))
    cached = []
    cap = min(SCORE_CAPS[loss], _loss_range(loss, (0., 300.)))
    for k in KS:
        for witness in witnesses:
            if witness == "unbiased_variance":
                raw = environment_mse_unbiased_scores(
                    environments, k=k, clip=(0., 300.))
                candidate_cap = min(cap, 300. ** 2 * k / (4 * (k - 1)))
            else:
                raw = environment_saa_scores(
                    environments, loss=loss, k=k, clip=(0., 300.))
                candidate_cap = cap
            cached.append((k, witness, np.minimum(raw, candidate_cap), candidate_cap))

    def route(methods):
        alpha = (.05 / len(LOSSES)) / (len(cached) * len(methods))
        candidates = []
        for k, witness, scores, candidate_cap in cached:
            for method in methods:
                candidates.append({
                    "k": k, "witness": witness, "method": method,
                    "bound": _lower_mean(scores, candidate_cap, alpha, method),
                    "n_blocks": len(scores), "score_cap": candidate_cap,
                })
        chosen = max(candidates, key=lambda row: row["bound"])
        return {**chosen, "methods": list(methods),
                "alpha_per_candidate": alpha, "candidates": candidates}

    return (route(METHODS), route(("dKelly_betting",)),
            route(METHODS + ("dKelly_betting",)))


def run(design, seed, n=800, n_environments=5_000):
    data = make(design, n, seed)
    environments = repeated_outcomes(
        data, n_environments, seed + 10_000_000, common_quantile=False)
    truth = gold.pointwise(data["fam"], data["s"], H=(15., 30., 60.))
    truth["Q50"] = .5 * truth["MAE"]
    legacy, betting, joint = {}, {}, {}
    for loss in LOSSES:
        legacy[loss], betting[loss], joint[loss] = compare_methods(
            environments, loss)
    return {
        "design": design, "seed": seed, "n": n,
        "n_environments": n_environments,
        "gold_original": {loss: float(np.mean(truth[loss])) for loss in LOSSES},
        "legacy_three": legacy, "dkelly_only": betting,
        "joint_four_method": joint,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=110001)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--environments", type=int, default=5_000)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1, 11):
            for seed in range(args.first_seed, args.first_seed + args.reps):
                result = run(design, seed, n_environments=args.environments)
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                print(design, seed, {
                    loss: (round(result["joint_four_method"][loss]["bound"], 4),
                           result["joint_four_method"][loss]["method"])
                    for loss in LOSSES}, flush=True)
