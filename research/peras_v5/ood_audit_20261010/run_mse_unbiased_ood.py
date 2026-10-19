"""Fresh audit of the unbiased repeated-environment MSE witness."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
import gold
from repeated_environment_saa import (
    METHODS,
    _lower_mean,
    environment_mse_unbiased_scores,
    environment_saa_portfolio,
    environment_saa_scores,
)
from run_coupled_environment_ood import repeated_outcomes


KS = (2, 4, 8, 16, 32)
CAP = 1000.
CLIP = (0., 300.)


def dedicated_route(environments, witness):
    alpha = .05 / (len(KS) * len(METHODS))
    rows = []
    for k in KS:
        if witness == "unbiased_variance":
            raw = environment_mse_unbiased_scores(environments, k=k, clip=CLIP)
            range_cap = CLIP[1] ** 2 * k / (4 * (k - 1))
            cap = min(CAP, range_cap)
        else:
            raw = environment_saa_scores(environments, loss="MSE", k=k, clip=CLIP)
            cap = CAP
        scores = np.minimum(raw, cap)
        for method in METHODS:
            rows.append({
                "witness": witness, "k": k, "method": method,
                "bound": _lower_mean(scores, cap, alpha, method),
                "score_mean": float(np.mean(scores)), "score_cap": cap,
            })
    return max(rows, key=lambda x: x["bound"])


def run(design, seed, n=800, n_environments=5_000):
    data = make(design, n, seed)
    environments = repeated_outcomes(
        data, n_environments, seed + 9_600_000, common_quantile=False)
    truth = float(np.mean(gold.pointwise(data["fam"], data["s"])["MSE"]))
    combined = environment_saa_portfolio(
        environments, loss="MSE", score_cap=CAP, ks=KS, clip=CLIP,
        alpha_family=.05,
        environment_provenance="registered coordinate-independent target environments")
    return {
        "design": design, "seed": seed, "n": n,
        "n_environments": n_environments, "gold_original": truth,
        "legacy_saa_only": dedicated_route(environments, "empirical_optimum"),
        "unbiased_only": dedicated_route(environments, "unbiased_variance"),
        "selection_safe_combined": combined,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=106001)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--environments", type=int, default=5_000)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1, 11):
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed, n_environments=args.environments)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(design, seed, row["legacy_saa_only"]["bound"],
                      row["unbiased_only"]["bound"],
                      row["selection_safe_combined"]["bound"], flush=True)
