"""Post-hoc numeric residual-W1 audit for learned environment alignment."""
import argparse
import json

import numpy as np

from audit_ood import make
from coupled_environment_w1 import environment_costs
import peras5 as P
from run_coupled_environment_ood import KS, ppf, repeated_outcomes
from run_disjoint_transport_audit import greedy_feature_blocks
from run_learned_alignment_environment_ood import RESIDUAL_CLIP


def run(path, grid_size=5_000):
    rows = [json.loads(line) for line in open(path)]
    u = (np.arange(grid_size) + .5) / grid_size
    gaps = []
    for row in rows:
        d = make(row["design"], row["n"], row["seed"])
        environments = repeated_outcomes(
            d, row["n_environments"], row["seed"] + 9_300_000,
            common_quantile=True)
        learn, _ = np.array_split(environments, 2, axis=0)
        shift = np.median(learn, axis=0)
        z = np.empty((grid_size, row["n"]))
        for fam in np.unique(d["fam"]):
            mask = d["fam"] == fam
            z[:, mask] = ppf(fam, u[:, None])
        coupled = d["mu"][None, :] + d["s"][None, :] * z
        representation = P.represent(d)
        for k in KS:
            blocks = greedy_feature_blocks(representation, k)
            numeric_w1 = float(np.mean(environment_costs(
                coupled, blocks, clip=RESIDUAL_CLIP, shift=shift)[0]))
            cap = row["learned_median_alignment"]["caps"][str(k)]["mean_w1_cap"]
            gaps.append(cap - numeric_w1)
    out = {
        "runs": len(rows), "grid_size": grid_size,
        "cap_checks": len(gaps),
        "numeric_population_cap_misses": sum(g < -2e-3 for g in gaps),
        "minimum_cap_minus_numeric_w1": float(min(gaps)),
        "median_cap_minus_numeric_w1": float(np.median(gaps)),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--grid-size", type=int, default=5_000)
    args = ap.parse_args()
    run(args.path, args.grid_size)
