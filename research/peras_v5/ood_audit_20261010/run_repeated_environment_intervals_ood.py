"""Fresh lower/upper attainable-risk intervals from repeated environments."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
from clipped_risk_gold import pointwise_risks
import gold
from repeated_environment_saa import (
    LOSSES,
    environment_holdout_upper,
    environment_saa_portfolio,
)
from run_coupled_environment_ood import repeated_outcomes


SCORE_CAPS = {"MAE": 30., "MSE": 1000., "Q10": 6., "Q50": 15., "Q90": 6.}
CLIP = (0., 300.)
KS = (2, 4, 8, 16, 32)


def run(design, seed, n=800, n_environments=5_000):
    d = make(design, n, seed)
    environments = repeated_outcomes(
        d, n_environments, seed + 9_700_000, common_quantile=False)
    clipped = pointwise_risks(d["fam"], d["mu"], d["s"], clip=CLIP)
    original = gold.pointwise(d["fam"], d["s"])
    original["Q50"] = .5 * original["MAE"]
    results = {}
    for loss in LOSSES:
        lower = environment_saa_portfolio(
            environments, loss=loss, score_cap=SCORE_CAPS[loss], ks=KS,
            clip=CLIP, alpha_family=.025 / len(LOSSES),
            environment_provenance="registered coordinate-independent target environments")
        upper = environment_holdout_upper(
            environments, loss=loss, clip=CLIP,
            alpha_family=.025 / len(LOSSES),
            environment_provenance="registered coordinate-independent target environments")
        results[loss] = {
            "clipped_gold": float(np.mean(clipped[loss])),
            "original_gold": float(np.mean(original[loss])),
            "lower": lower,
            "upper": upper,
            "interval_width": float(upper["upper"] - lower["bound"]),
        }
    return {"design": design, "seed": seed, "n": n,
            "n_environments": n_environments, "clip": CLIP,
            "losses": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=105001)
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
                    loss: [row["losses"][loss]["lower"]["bound"],
                           row["losses"][loss]["upper"]["upper"]]
                    for loss in LOSSES}, flush=True)
