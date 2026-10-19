"""Fresh direct attainable-risk audit using repeated target environments."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
import gold
from repeated_environment_saa import LOSSES, environment_saa_portfolio
from run_coupled_environment_ood import repeated_outcomes


SCORE_CAPS = {"MAE": 30., "MSE": 1000., "Q10": 6., "Q50": 15., "Q90": 6.}
CLIP = (0., 300.)
KS = (2, 4, 8, 16, 32)


def route(environments, provenance):
    return {loss: environment_saa_portfolio(
        environments, loss=loss, score_cap=SCORE_CAPS[loss], ks=KS,
        clip=CLIP, alpha_family=.05 / len(LOSSES),
        environment_provenance=provenance) for loss in LOSSES}


def run(design, seed, n=800, n_environments=5_000):
    d = make(design, n, seed)
    common = repeated_outcomes(
        d, n_environments, seed + 9_400_000, common_quantile=True)
    product = repeated_outcomes(
        d, n_environments, seed + 9_500_000, common_quantile=False)
    truth = gold.pointwise(d["fam"], d["s"])
    truth["Q50"] = .5 * truth["MAE"]
    return {
        "design": design, "seed": seed, "n": n,
        "n_environments": n_environments, "clip": CLIP,
        "gold_original": {loss: float(np.mean(truth[loss])) for loss in LOSSES},
        "common_quantile": route(
            common, "registered common-quantile target environments"),
        "product_coupling": route(
            product, "registered coordinate-independent target environments"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=103001)
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
                print(design, seed, {
                    loss: row["product_coupling"][loss]["bound"] for loss in LOSSES
                }, flush=True)
