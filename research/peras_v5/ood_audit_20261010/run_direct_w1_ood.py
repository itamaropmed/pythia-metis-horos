"""Fresh exact-law OOD audit of direct-average W1 calibration.

The group-law equality key is known from this synthetic generator. Both
calibration and audit outcomes are independently generated. No prediction
model is trained; no synthetic oracle W1 enters either bound.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from disjoint_transport_bound import lower_bound
from independent_w1_calibration import calibrate_mean_block_w1_direct
from run_exact_law_battery import DESIGNS, make
import gold


CLIP = (0.0, 300.0)
SCORE_CLIPS = {"MAE": 30.0, "Q90": 6.0, "MSE": 1000.0}
CAL_DELTA = 0.025
EVAL_ALPHA = 0.025 / len(SCORE_CLIPS)


def run(design: int, seed: int, n_cal: int = 20_000,
        n_audit: int = 800) -> dict:
    cal = make(design, n_cal, seed + 2_000_000)
    audit = make(design, n_audit, seed)
    samples = defaultdict(list)
    for i in range(n_cal):
        key = (int(cal["code"][i]), int(cal["concepts"][i, 0]))
        samples[key].append(float(cal["y"][i]))
    groups = np.empty(n_audit, dtype=object)
    for i in range(n_audit):
        groups[i] = (int(audit["code"][i]), int(audit["concepts"][i, 0]))
    # Freeze this outcome-blind block construction: adjacent sorted codes
    # within each concept. All 800 audit entities enter exactly once.
    order = np.lexsort((np.arange(n_audit), audit["code"],
                        audit["concepts"][:, 0]))
    blocks = order.reshape(n_audit // 2, 2)
    calibrated = calibrate_mean_block_w1_direct(
        samples, groups, blocks, clip=CLIP, delta=CAL_DELTA)
    truth = gold.pointwise(audit["fam"], audit["s"])
    result = {"design": design, "seed": seed, "n_cal": n_cal,
              "n_audit": n_audit, "n_groups": len(samples),
              "direct_cap": calibrated["mean_w1_cap"],
              "dkw_cap": calibrated["pairwise_dkw_mean_cap"],
              "empirical_w1": calibrated["empirical_mean_w1"],
              "direct_radius": calibrated["concentration_radius"],
              "cross_group_weight": calibrated["cross_group_pair_weight"],
              "gold": {}, "direct_bound": {}, "dkw_bound": {}}
    for loss, score_clip in SCORE_CLIPS.items():
        result["gold"][loss] = float(np.mean(truth[loss]))
        common = dict(loss=loss, score_clip=score_clip,
                      alpha=EVAL_ALPHA, outcome_clip=CLIP)
        result["direct_bound"][loss] = lower_bound(
            audit["y"], blocks, mean_w1_cap=result["direct_cap"],
            cap_provenance="independent direct average-W1 calibration",
            **common)["bound"]
        result["dkw_bound"][loss] = lower_bound(
            audit["y"], blocks, mean_w1_cap=result["dkw_cap"],
            cap_provenance="independent simultaneous pairwise DKW calibration",
            **common)["bound"]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=90001)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output path already exists")
    with args.out.open("w") as f:
        for design in DESIGNS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(design, seed, row["direct_cap"],
                      row["dkw_cap"], flush=True)


if __name__ == "__main__":
    main()
