"""Fresh near-replicate audit using outcome-only repeated-environment W1 caps."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
from coupled_environment_w1 import calibrate_coupled_mean_w1
from disjoint_transport_bound import lower_bound
import gold
import peras5 as P
from run_disjoint_transport_audit import greedy_feature_blocks


KS = (2, 4, 8)
LOSSES = ("MAE", "Q90", "MSE")
SCORE_CLIPS = {"MAE": (10., 20., 40.),
               "Q90": (2., 5., 10.),
               "MSE": (200., 400., 800.)}
OUTCOME_CLIP = (0., 300.)
CAP_DELTA_FAMILY = .025 / len(KS)
BOUND_ALPHA = .025 / (len(KS) * len(LOSSES) * 3)


_CDF = {fam: gold.FAMILIES[fam](gold.GRID) for fam in gold.FAMILIES}


def ppf(fam, u):
    return np.interp(u, _CDF[str(fam)], gold.GRID)


def repeated_outcomes(d, n_environments, seed, *, common_quantile):
    rng = np.random.default_rng(seed)
    n = len(d["mu"])
    u = (rng.random(n_environments)[:, None] if common_quantile
         else rng.random((n_environments, n)))
    z = np.empty((n_environments, n))
    for fam in np.unique(d["fam"]):
        mask = d["fam"] == fam
        source = u if common_quantile else u[:, mask]
        values = ppf(fam, source)
        z[:, mask] = values if not common_quantile else values
    return d["mu"][None, :] + d["s"][None, :] * z


def route(d, replicates, blocks_by_k, provenance):
    caps = {}
    candidates = {loss: [] for loss in LOSSES}
    for k, blocks in blocks_by_k.items():
        calibrated = calibrate_coupled_mean_w1(
            replicates, blocks, clip=OUTCOME_CLIP,
            delta_family=CAP_DELTA_FAMILY,
            coupling_provenance=provenance)
        caps[k] = {
            "mean_w1_cap": calibrated["mean_w1_cap"],
            "empirical_coupling_cost": calibrated["empirical_coupling_cost"],
            "selected_method": calibrated["selected_method"],
            "environment_cost_sd": calibrated["environment_cost_sd"],
        }
        for loss in LOSSES:
            for score_clip in SCORE_CLIPS[loss]:
                result = lower_bound(
                    d["y"], blocks, loss=loss,
                    mean_w1_cap=calibrated["mean_w1_cap"],
                    score_clip=score_clip, alpha=BOUND_ALPHA,
                    cap_provenance=provenance,
                    outcome_clip=OUTCOME_CLIP)
                candidates[loss].append({**result, "k": k})
    return {"caps": caps,
            "chosen": {loss: max(candidates[loss], key=lambda row: row["bound"])
                       for loss in LOSSES},
            "positive_candidates": {loss: sum(row["bound"] > 0
                                                for row in candidates[loss])
                                    for loss in LOSSES}}


def run(design, seed, n=800, n_environments=1_000):
    d = make(design, n, seed)
    representation = P.represent(d)
    blocks = {k: greedy_feature_blocks(representation, k) for k in KS}
    common = repeated_outcomes(
        d, n_environments, seed + 9_100_000, common_quantile=True)
    independent = repeated_outcomes(
        d, n_environments, seed + 9_200_000, common_quantile=False)
    truth = gold.pointwise(d["fam"], d["s"])
    return {
        "design": design, "seed": seed, "n": n,
        "n_environments": n_environments, "outcome_clip": OUTCOME_CLIP,
        "gold_original": {loss: float(np.mean(truth[loss])) for loss in LOSSES},
        "common_quantile": route(
            d, common, blocks,
            "independent registered common-quantile synthetic environments"),
        "independent_coupling": route(
            d, independent, blocks,
            "independent registered product-coupled synthetic environments"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=101001)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--environments", type=int, default=1_000)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1, 11):
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed, n_environments=args.environments)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(design, seed,
                      row["common_quantile"]["chosen"]["MAE"]["bound"],
                      row["independent_coupling"]["chosen"]["MAE"]["bound"],
                      flush=True)
