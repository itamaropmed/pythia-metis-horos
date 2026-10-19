"""Selection-safe disjoint-pair bounds for a synthetic exact-law design.

This is an accepted-method comparator, not a new Peras theorem. The equality
key comes from the DGP and is not available for Mayo without external proof.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_ood import make
import gold


LOSSES = ("MAE", "Q90", "MSE")
CLIPS = {"MAE": (5.0, 10.0, 20.0, 40.0, 80.0),
         "Q90": (1.0, 2.0, 5.0, 10.0, 20.0),
         "MSE": (50.0, 100.0, 200.0, 400.0, 800.0)}
METHODS = ("Hoeffding", "empirical_Bernstein", "Bentkus")
ALPHA = 0.05 / (len(LOSSES) * len(CLIPS["MAE"]) * len(METHODS))


def exact_pairs(d: dict) -> np.ndarray:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (code, complexity) in enumerate(zip(d["code"], d["concepts"][:, 0])):
        groups[(int(code), int(complexity))].append(i)
    pairs = []
    for group in groups.values():
        pairs.extend(zip(group[::2], group[1::2]))
    return np.asarray(pairs, dtype=int)


def lower_mean(scores: np.ndarray, cap: float, method: str,
               alpha: float = ALPHA) -> float:
    from peras.core.concentration import bentkus_lower_mean

    n = len(scores)
    if method == "Bentkus":
        return bentkus_lower_mean(scores, cap, alpha)
    if method == "Hoeffding":
        radius = cap * np.sqrt(np.log(1.0 / alpha) / (2.0 * n))
    elif method == "empirical_Bernstein":
        log_term = np.log(2.0 / alpha)
        radius = (np.sqrt(2.0 * np.var(scores, ddof=1) * log_term / n)
                  + 7.0 * cap * log_term / (3.0 * (n - 1)))
    else:
        raise ValueError(method)
    return max(0.0, float(np.mean(scores) - radius))


def run(seed: int, n: int = 800) -> dict:
    d = make(10, n, seed)
    pairs = exact_pairs(d)
    if pairs.shape[0] < 2:
        raise ValueError("too few exact pairs")
    if np.unique(pairs).size != pairs.size:
        raise AssertionError("matching reused an entity")
    left, right = pairs.T
    if (not np.array_equal(d["mu"][left], d["mu"][right])
            or not np.array_equal(d["s"][left], d["s"][right])
            or not np.array_equal(d["fam"][left], d["fam"][right])):
        raise AssertionError("the synthetic equality key did not identify equal conditional laws")
    diff = d["y"][pairs[:, 0]] - d["y"][pairs[:, 1]]
    witnesses = {"MAE": 0.5 * np.abs(diff),
                 "Q90": 0.5 * min(0.9, 0.1) * np.abs(diff),
                 "MSE": 0.5 * diff ** 2}
    fraction = 2 * len(pairs) / n
    truth = gold.pointwise(d["fam"], d["s"])
    chosen = {}
    for loss in LOSSES:
        options = []
        for cap in CLIPS[loss]:
            clipped = np.minimum(witnesses[loss], cap)
            for method in METHODS:
                options.append({"method": method, "cap": cap,
                                "bound": fraction * lower_mean(clipped, cap, method)})
        chosen[loss] = max(options, key=lambda x: x["bound"])
    return {"design": 10, "seed": seed, "n": n,
            "n_pairs": len(pairs), "fraction": fraction,
            "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
            "chosen": chosen}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=30001)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    for seed in range(args.seed_start, args.seed_start + args.reps):
        row = run(seed)
        with args.out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"seed": seed, "n_pairs": row["n_pairs"],
                          "chosen": row["chosen"]}), flush=True)
