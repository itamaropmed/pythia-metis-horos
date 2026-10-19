"""Resource-matched fresh-seed grouped SAA versus transport audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from group_calibration_floor import grouped_floor
from run_direct_w1_ood import CLIP, SCORE_CLIPS, run as transport_run
from run_exact_law_battery import DESIGNS, make


def run(design: int, seed: int) -> dict:
    row = transport_run(design, seed)
    cal = make(design, row["n_cal"], seed + 2_000_000)
    audit = make(design, row["n_audit"], seed)
    samples = defaultdict(list)
    for code, concept, y in zip(cal["code"], cal["concepts"][:, 0], cal["y"]):
        samples[(int(code), int(concept))].append(float(y))
    counts = Counter((int(code), int(concept)) for code, concept
                     in zip(audit["code"], audit["concepts"][:, 0]))
    weights = {g: n / row["n_audit"] for g, n in counts.items()}
    row["grouped_saa"] = {loss: grouped_floor(
        samples, weights, loss=loss, clip=CLIP, score_clip=score_clip,
        k=8, alpha=0.05 / len(SCORE_CLIPS))
        for loss, score_clip in SCORE_CLIPS.items()}
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=91001)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists")
    with args.out.open("w") as f:
        for design in DESIGNS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(design, seed, row["direct_bound"]["MAE"],
                      row["grouped_saa"]["MAE"]["tuple_bound"],
                      row["grouped_saa"]["MAE"]["full_saa_bound"], flush=True)
