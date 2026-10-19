"""Joint 95% pair-or-tuple conditional lower bound on exact-law batteries.

All 225 choices (3 losses × 3 concentration methods × 5 caps × 5 structures)
are Bonferroni allocated before viewing the fresh 34001–34005 seed block.
Conditional coverage still requires the supplied grouping to be truly exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gold
from run_disjoint_tuple_concentration import KS
from run_exact_law_battery import DESIGNS, make, pair_route, tuple_route
from run_exact_pair_concentration import CLIPS, LOSSES, METHODS


N_CANDIDATES = len(LOSSES) * len(CLIPS["MAE"]) * len(METHODS) * (1 + len(KS))
ALPHA = 0.05 / N_CANDIDATES


def run(design: int, seed: int, n: int = 800) -> dict:
    d = make(design, n, seed)
    truth = gold.pointwise(d["fam"], d["s"])
    pairs = pair_route(d, alpha=ALPHA)
    tuples = tuple_route(d, alpha=ALPHA)
    selected = {}
    for loss in LOSSES:
        route = "tuple" if tuples[loss]["bound"] > pairs[loss]["bound"] else "pair"
        selected[loss] = {"route": route,
                          "bound": max(pairs[loss]["bound"], tuples[loss]["bound"])}
    return {"design": design, "seed": seed, "n": n,
            "claim_tier": "conditional", "assumption": "oracle exact-law equality by (code, concept)",
            "family_alpha": 0.05, "candidates": N_CANDIDATES,
            "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
            "pair": pairs, "tuple": tuples, "selected": selected}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=34001)
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
                              "selected_Q90": row["selected"]["Q90"]}), flush=True)
