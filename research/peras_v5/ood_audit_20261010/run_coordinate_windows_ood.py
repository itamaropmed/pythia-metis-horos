"""Fresh product-coordinate audit for W15/W30/W60 miss risk."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
import gold
from repeated_environment_saa import (
    environment_coordinate_saa_portfolio,
    environment_saa_portfolio,
)
from run_coupled_environment_ood import repeated_outcomes


LOSSES = ("W15", "W30", "W60")
KS = (2, 4, 8, 16, 32)


def run(design, seed, n=800, n_environments=5_000):
    data = make(design, n, seed)
    environments = repeated_outcomes(
        data, n_environments, seed + 9_900_000, common_quantile=False)
    truth = gold.pointwise(data["fam"], data["s"], H=(15., 30., 60.))
    row, coordinate = {}, {}
    for loss in LOSSES:
        row[loss] = environment_saa_portfolio(
            environments, loss=loss, score_cap=1., ks=KS,
            clip=(0., 300.), alpha_family=.05 / len(LOSSES),
            environment_provenance="registered coordinate-independent target environments")
        coordinate[loss] = environment_coordinate_saa_portfolio(
            environments, loss=loss, score_cap=1., ks=KS,
            clip=(0., 300.), alpha_family=.05 / len(LOSSES),
            environment_provenance="registered coordinate-independent target environments",
            coordinate_independence_provenance="product RNG streams across registered target units")
    return {
        "design": design, "seed": seed, "n": n,
        "n_environments": n_environments,
        "gold_original": {loss: float(np.mean(truth[loss])) for loss in LOSSES},
        "row_level": row, "coordinate_independent": coordinate,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=109001)
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
                    loss: round(result["coordinate_independent"][loss]["bound"], 4)
                    for loss in LOSSES}, flush=True)
