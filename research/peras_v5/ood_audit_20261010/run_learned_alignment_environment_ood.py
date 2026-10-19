"""Fresh sample-split learned alignment plus coupled-environment W1 audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
from coupled_environment_w1 import calibrate_coupled_mean_w1
from disjoint_transport_bound import lower_bound
import gold
import peras5 as P
from run_coupled_environment_ood import (
    KS,
    LOSSES,
    SCORE_CLIPS,
    repeated_outcomes,
)
from run_disjoint_transport_audit import greedy_feature_blocks


RESIDUAL_CLIP = (-150., 150.)
CAP_DELTA_FAMILY = .025 / len(KS)
BOUND_ALPHA = .025 / (len(KS) * len(LOSSES) * 3)


def aligned_route(d, environments, blocks_by_k):
    if len(environments) < 4 or len(environments) % 2:
        raise ValueError("an even number of at least four environments is required")
    learn, calibrate = np.array_split(environments, 2, axis=0)
    shift = np.median(learn, axis=0)
    caps = {}
    candidates = {loss: [] for loss in LOSSES}
    for k, blocks in blocks_by_k.items():
        cap = calibrate_coupled_mean_w1(
            calibrate, blocks, clip=RESIDUAL_CLIP,
            delta_family=CAP_DELTA_FAMILY,
            coupling_provenance="held-out common-quantile synthetic environments",
            shift=shift,
            shift_provenance="per-unit median from disjoint environment-learning split")
        caps[k] = {
            "mean_w1_cap": cap["mean_w1_cap"],
            "empirical_coupling_cost": cap["empirical_coupling_cost"],
            "selected_method": cap["selected_method"],
            "environment_cost_sd": cap["environment_cost_sd"],
        }
        for loss in LOSSES:
            for score_clip in SCORE_CLIPS[loss]:
                result = lower_bound(
                    d["y"], blocks, loss=loss,
                    mean_w1_cap=cap["mean_w1_cap"],
                    score_clip=score_clip, alpha=BOUND_ALPHA,
                    cap_provenance="sample-split repeated-environment residual W1 cap",
                    outcome_clip=RESIDUAL_CLIP,
                    shift=shift,
                    shift_provenance="per-unit median from disjoint environment-learning split")
                candidates[loss].append({**result, "k": k})
    return {
        "n_shift_learning_environments": len(learn),
        "n_cap_calibration_environments": len(calibrate),
        "caps": caps,
        "shift_summary": {
            "minimum": float(np.min(shift)), "median": float(np.median(shift)),
            "maximum": float(np.max(shift)), "sd": float(np.std(shift)),
        },
        "chosen": {loss: max(candidates[loss], key=lambda row: row["bound"])
                   for loss in LOSSES},
        "positive_candidates": {loss: sum(row["bound"] > 0
                                            for row in candidates[loss])
                                for loss in LOSSES},
    }


def run(design, seed, n=800, n_environments=5_000):
    d = make(design, n, seed)
    representation = P.represent(d)
    blocks = {k: greedy_feature_blocks(representation, k) for k in KS}
    environments = repeated_outcomes(
        d, n_environments, seed + 9_300_000, common_quantile=True)
    truth = gold.pointwise(d["fam"], d["s"])
    return {
        "design": design, "seed": seed, "n": n,
        "n_environments": n_environments,
        "residual_clip": RESIDUAL_CLIP,
        "gold_original": {loss: float(np.mean(truth[loss])) for loss in LOSSES},
        "learned_median_alignment": aligned_route(d, environments, blocks),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=102001)
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
                    loss: row["learned_median_alignment"]["chosen"][loss]["bound"]
                    for loss in LOSSES}, flush=True)
