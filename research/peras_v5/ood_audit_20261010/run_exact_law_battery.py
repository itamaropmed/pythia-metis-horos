"""Untouched exact-replicate noise-law battery for pair vs disjoint tuples.

All candidate grids are imported unchanged from the preceding design-10 audit.
The generator varies noise families, heteroscedasticity, and cell density.
The oracle exact-law key is (code, concept); this is not a Mayo grouping rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core import erm_risk
import gold
from run_disjoint_tuple_concentration import KS, ALPHA as TUPLE_ALPHA, exact_blocks
from run_exact_pair_concentration import (
    ALPHA as PAIR_ALPHA, CLIPS, LOSSES, METHODS, exact_pairs, lower_mean,
)


DESIGNS = {
    1: (10, "normal", "constant"),
    2: (40, "laplace", "constant"),
    3: (80, "gamma2", "constant"),
    4: (20, "lognorm", "constant"),
    5: (40, "t3", "constant"),
    6: (60, "t5", "hetero"),
    7: (40, "mix05", "hetero"),
    8: (40, "mix35", "constant"),
    9: (20, "normal_laplace", "hetero"),
    10: (80, "gamma_lognorm", "hetero"),
}


def make(design: int, n: int, seed: int) -> dict:
    cells, family, scale_type = DESIGNS[design]
    rng = np.random.default_rng(seed + 10007 * design)
    code = rng.integers(0, cells, n)
    c = rng.choice(3, n, p=(0.45, 0.35, 0.20))
    mu = 75.0 + 2.3 * code + 13.0 * c
    scale = 20.0 if scale_type == "constant" else 10.0 + 4.0 * (code % 6)
    s = np.broadcast_to(scale, (n,)).astype(float).copy()
    if family == "normal_laplace":
        fam = np.where(code % 2 == 0, "normal", "laplace")
    elif family == "gamma_lognorm":
        fam = np.where(code % 2 == 0, "gamma2", "lognorm")
    else:
        fam = np.full(n, family, dtype=object)
    z = np.empty(n)
    for f in np.unique(fam):
        mask = fam == f
        z[mask] = gold.sample(f, int(mask.sum()), rng)
    notes = np.where(c == 0, "routine complexity", np.where(c == 1,
                      "moderate complexity", "high complexity"))
    return {"code": code, "concepts": c[:, None], "notes": notes,
            "mu": mu, "s": s, "fam": fam, "y": mu + s * z}


def pair_route(d: dict, alpha: float = PAIR_ALPHA) -> dict:
    pairs = exact_pairs(d)
    if pairs.shape[0] < 2 or np.unique(pairs).size != pairs.size:
        raise AssertionError("pair matching failed")
    left, right = pairs.T
    for key in ("mu", "s", "fam"):
        if not np.array_equal(d[key][left], d[key][right]):
            raise AssertionError("oracle equality key failed")
    diff = d["y"][left] - d["y"][right]
    witnesses = {"MAE": 0.5 * np.abs(diff),
                 "Q90": 0.05 * np.abs(diff),
                 "MSE": 0.5 * diff ** 2}
    fraction = 2 * len(pairs) / len(d["y"])
    out = {}
    for loss in LOSSES:
        opts = []
        for cap in CLIPS[loss]:
            score = np.minimum(witnesses[loss], cap)
            for method in METHODS:
                opts.append({"bound": fraction * lower_mean(score, cap, method, alpha),
                             "cap": cap, "method": method, "n_pairs": len(pairs)})
        out[loss] = max(opts, key=lambda row: row["bound"])
    return out


def tuple_route(d: dict, alpha: float = TUPLE_ALPHA) -> dict:
    options = {loss: [] for loss in LOSSES}
    n = len(d["y"])
    for k in KS:
        blocks = exact_blocks(d, k)
        if blocks.shape[0] < 2 or np.unique(blocks).size != blocks.size:
            raise AssertionError("tuple partition failed")
        for key in ("mu", "s", "fam"):
            if not np.all(d[key][blocks] == d[key][blocks[:, :1]]):
                raise AssertionError("oracle equality key failed")
        y_block = d["y"][blocks]
        fraction = blocks.size / n
        for loss in LOSSES:
            witness = erm_risk(loss, y_block)
            if loss == "MSE":
                witness = witness * k / (k - 1)
            for cap in CLIPS[loss]:
                score = np.minimum(witness, cap)
                for method in METHODS:
                    options[loss].append({
                        "bound": fraction * lower_mean(score, cap, method, alpha),
                        "cap": cap, "method": method, "tuple_size": k,
                        "n_blocks": len(blocks),
                    })
    return {loss: max(options[loss], key=lambda row: row["bound"]) for loss in LOSSES}


def run(design: int, seed: int, n: int = 800) -> dict:
    d = make(design, n, seed)
    truth = gold.pointwise(d["fam"], d["s"])
    return {"design": design, "seed": seed, "n": n,
            "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
            "pair": pair_route(d), "tuple": tuple_route(d)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=33001)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    for design in DESIGNS:
        for seed in range(args.seed_start, args.seed_start + args.reps):
            row = run(design, seed)
            with args.out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps({"design": design, "seed": seed,
                              "pair_Q90": row["pair"]["Q90"]["bound"],
                              "tuple_Q90": row["tuple"]["Q90"]["bound"]}), flush=True)
