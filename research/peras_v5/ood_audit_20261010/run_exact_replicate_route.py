"""Outcome-blind exact-law grouping in OOD design 10; synthetic-only diagnostic.

The generating law in design 10 depends exactly on (code, note complexity).
That fact is supplied by the DGP, not learned from the outcome sample.  It is
unavailable for Mayo absent an independently justified equality rule.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_ood import make
import gold
from transport_tuple_bound import lower_bound


KS = (2, 4, 6)
CLIPS = ((40.0, 280.0), (70.0, 230.0), (90.0, 210.0))
LOSSES = ("MAE", "Q90", "MSE")
ALPHA = 0.05 / (len(KS) * len(CLIPS) * len(LOSSES))


def exact_tuples(d: dict, k: int) -> np.ndarray:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (code, complexity) in enumerate(zip(d["code"], d["concepts"][:, 0])):
        groups[(int(code), int(complexity))].append(i)
    rows = []
    for group in groups.values():
        if len(group) < k:
            continue
        for offset, anchor in enumerate(group):
            rows.append([anchor] + [group[(offset + j) % len(group)] for j in range(1, k)])
    return np.asarray(rows, dtype=int)


def run(seed: int, n: int = 800) -> dict:
    d = make(10, n, seed)
    truth = gold.pointwise(d["fam"], d["s"])
    candidates = {loss: [] for loss in LOSSES}
    for k in KS:
        idx = exact_tuples(d, k)
        if idx.size == 0:
            continue
        zero = np.zeros_like(idx, dtype=float)
        fraction = idx.shape[0] / n
        for clip in CLIPS:
            for loss in LOSSES:
                b = lower_bound(
                    d["y"], idx, zero, loss=loss, clip=clip,
                    alpha=ALPHA,
                    cap_provenance="synthetic DGP equality by (code, note complexity)",
                )
                candidates[loss].append({
                    "bound": fraction * b["bound"],
                    "subset_bound": b["bound"],
                    "fraction": fraction,
                    "tuple_size": k,
                    "clip": list(clip),
                    "statistic": b["statistic"],
                    "radius": b["concentration_radius"],
                })
    return {
        "design": 10, "seed": seed, "n": n,
        "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
        "chosen": {loss: max(candidates[loss], key=lambda row: row["bound"])
                   for loss in LOSSES},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=29001)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    for seed in range(args.seed_start, args.seed_start + args.reps):
        row = run(seed)
        with args.out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"seed": seed, "gold": row["gold"],
                          "bound": {loss: row["chosen"][loss]["bound"] for loss in LOSSES}}),
              flush=True)
