"""Simultaneous clipping/tuple-size portfolio on untouched outcome seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
import gold
import peras5 as P
from run_transport_audit import oracle_transport_caps
from transport_tuple_bound import lower_bound


CLIPS = ((40.0, 280.0), (70.0, 230.0), (90.0, 210.0))
KS = (4, 8, 12)
LOSSES = ("MAE", "Q90", "MSE")
ALPHA = 0.05 / (len(CLIPS) * len(KS) * len(LOSSES))


def run(design: int, seed: int, n: int = 800) -> dict:
    d = make(design, n, seed)
    Z = P.represent(d)
    _, nb = P.neighbours(Z, max(KS) - 1)
    candidates = {loss: [] for loss in LOSSES}
    for k in KS:
        idx = np.column_stack([np.arange(n), nb[:, :k - 1]])
        cap = oracle_transport_caps(d, idx)
        for clip in CLIPS:
            for loss in LOSSES:
                b = lower_bound(d["y"], idx, cap, loss=loss, clip=clip,
                                alpha=ALPHA,
                                cap_provenance="oracle analytic coupling from synthetic DGP")
                candidates[loss].append(b)
    truth = gold.pointwise(d["fam"], d["s"])
    chosen = {loss: max(candidates[loss], key=lambda b: b["bound"]) for loss in LOSSES}
    return {
        "design": design, "seed": seed, "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
        "chosen": {loss: {key: b[key] for key in ("bound", "clip", "tuple_size", "statistic",
                                                    "transport_penalty", "concentration_radius")}
                   for loss, b in chosen.items()},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=28001)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"output already exists: {args.out}; choose a new file")
    for design in range(1, 11):
        for seed in range(args.seed_start, args.seed_start + args.reps):
            row = run(design, seed)
            with args.out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"design": design, "seed": seed,
                              "MAE_bound": row["chosen"]["MAE"]["bound"]}), flush=True)
