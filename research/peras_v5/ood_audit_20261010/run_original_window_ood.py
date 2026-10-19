"""Fresh joint clipped-multiloss plus original-window interval audit."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from finite_group_multiloss_interval import LOSSES, SCORE_CAPS, interval
from run_exact_law_battery import DESIGNS, make
from tail_transfer import tail_upper, transfer_interval
import gold

ALPHA_EVENT = .05 / 19  # 16 clipped endpoints + W15/W30/W60 tail caps


def run(design, seed):
    calibration = make(design, 20_000, seed + 2_000_000)
    audit = make(design, 800, seed)
    samples = defaultdict(list)
    for code, concept, y in zip(calibration["code"], calibration["concepts"][:, 0],
                                calibration["y"]):
        samples[(int(code), int(concept))].append(float(y))
    groups = [(int(code), int(concept)) for code, concept
              in zip(audit["code"], audit["concepts"][:, 0])]
    weights = {g: n / 800 for g, n in Counter(groups).items()}
    clipped = {loss: interval(samples, weights, loss=loss,
                              score_cap=SCORE_CAPS[loss],
                              alpha_endpoint=ALPHA_EVENT) for loss in LOSSES}
    tails, original_window = {}, {}
    for loss in ("W15", "W30", "W60"):
        tails[loss] = tail_upper(samples, weights, loss=loss,
                                 clip=(0, 300), alpha=ALPHA_EVENT)
        original_window[loss] = transfer_interval(clipped[loss], tails[loss])
    truth = gold.pointwise(audit["fam"], audit["s"], H=(15., 30., 60.))
    original_gold = {loss: float(np.mean(truth[loss]))
                     for loss in ("W15", "W30", "W60")}
    return {"design": design, "seed": seed, "original_window_gold": original_gold,
            "clipped_intervals": clipped, "tail_caps": tails,
            "original_window_intervals": original_window,
            "joint_claim": "unconditional >=95% over 16 clipped endpoints and 3 tail caps, conditional on group/stationarity/independence assumptions"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=96001)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists")
    with args.out.open("w") as f:
        for design in DESIGNS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(design, seed, flush=True)
