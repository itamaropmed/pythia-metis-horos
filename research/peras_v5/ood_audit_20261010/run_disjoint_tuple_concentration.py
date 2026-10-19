"""Outcome-blind disjoint tuples with independent-score concentration.

This synthetic audit assumes the DGP's exact equality key. It is a test of
finite-tuple optimism against lost effective sample size, not a Mayo method.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_ood import make
from core import erm_risk
import gold
from run_exact_pair_concentration import CLIPS, LOSSES, METHODS, lower_mean


KS = (2, 3, 4, 6)
ALPHA = 0.05 / (len(LOSSES) * len(KS) * len(CLIPS["MAE"]) * len(METHODS))


def exact_blocks(d: dict, k: int) -> np.ndarray:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (code, complexity) in enumerate(zip(d["code"], d["concepts"][:, 0])):
        groups[(int(code), int(complexity))].append(i)
    blocks = []
    for group in groups.values():
        blocks.extend(group[start:start + k] for start in range(0, len(group) - k + 1, k))
    return np.asarray(blocks, dtype=int)


def run(seed: int, n: int = 800) -> dict:
    d = make(10, n, seed)
    truth = gold.pointwise(d["fam"], d["s"])
    candidates = {loss: [] for loss in LOSSES}
    for k in KS:
        blocks = exact_blocks(d, k)
        if blocks.shape[0] < 2 or np.unique(blocks).size != blocks.size:
            raise AssertionError("need at least two disjoint blocks")
        anchor = blocks[:, :1]
        if (not np.all(d["mu"][blocks] == d["mu"][anchor])
                or not np.all(d["s"][blocks] == d["s"][anchor])
                or not np.all(d["fam"][blocks] == d["fam"][anchor])):
            raise AssertionError("synthetic equality key failed")
        y_block = d["y"][blocks]
        fraction = blocks.size / n
        for loss in LOSSES:
            witness = erm_risk(loss, y_block)
            if loss == "MSE":
                # Exact-law iid sample variance: E[k/(k-1) * SAA] = Var(Y).
                witness = witness * k / (k - 1)
            for cap in CLIPS[loss]:
                clipped = np.minimum(witness, cap)
                for method in METHODS:
                    candidates[loss].append({
                        "bound": fraction * lower_mean(clipped, cap, method, ALPHA),
                        "tuple_size": k, "n_blocks": len(blocks),
                        "fraction": fraction, "cap": cap, "method": method,
                    })
    return {"design": 10, "seed": seed, "n": n,
            "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
            "chosen": {loss: max(candidates[loss], key=lambda row: row["bound"])
                       for loss in LOSSES}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=32001)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    for seed in range(args.seed_start, args.seed_start + args.reps):
        row = run(seed)
        with args.out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"seed": seed, "chosen": row["chosen"]}), flush=True)
