"""Fresh OOD audit of external pre-outcome translation alignment.

The zero/booked portfolio is deployable in principle only if its W1 caps are
independently established. Here all caps are synthetic-oracle diagnostics.
The conditional-mean alignment is an explicitly nondeployable ceiling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
from disjoint_transport_bound import lower_bound
import gold
import peras5 as P
from run_disjoint_transport_audit import (
    KS, LOSSES, SCORE_CLIPS, greedy_feature_blocks,
    oracle_mean_within_block_w1,
)


AVAILABLE_SHIFTS = ("zero", "booked")
AVAILABLE_ALPHA = 0.05 / (
    len(AVAILABLE_SHIFTS) * len(KS) * len(LOSSES) * len(SCORE_CLIPS["MAE"])
)
ORACLE_ALPHA = 0.05 / (len(KS) * len(LOSSES) * len(SCORE_CLIPS["MAE"]))
MSE_CLIPS = {"zero": (40.0, 280.0),
             "booked": (-160.0, 160.0),
             "oracle_mu": (-120.0, 120.0)}


def run(design: int, seed: int, n: int = 800) -> dict:
    d = make(design, n, seed)
    Z = P.represent(d)
    truth = gold.pointwise(d["fam"], d["s"])
    shifts = {"zero": np.zeros(n), "booked": d["tab"][:, 0],
              "oracle_mu": d["mu"]}
    candidates = {name: {loss: [] for loss in LOSSES} for name in shifts}
    for k in KS:
        blocks = greedy_feature_blocks(Z, k)
        for name, shift in shifts.items():
            residual_dgp = dict(d)
            residual_dgp["mu"] = d["mu"] - shift
            cap = oracle_mean_within_block_w1(residual_dgp, blocks)
            for loss in LOSSES:
                for score_clip in SCORE_CLIPS[loss]:
                    row = lower_bound(
                        d["y"], blocks, loss=loss, mean_w1_cap=cap,
                        score_clip=score_clip,
                        alpha=ORACLE_ALPHA if name == "oracle_mu" else AVAILABLE_ALPHA,
                        cap_provenance="analytic upper coupling from known synthetic residual laws",
                        outcome_clip=MSE_CLIPS[name] if loss == "MSE" else None,
                        shift=None if name == "zero" else shift,
                        shift_provenance=None if name == "zero" else
                                         ("pre-outcome booked slot" if name == "booked" else
                                          "known synthetic conditional mean; oracle only"),
                    )
                    candidates[name][loss].append(row)
    best = {name: {loss: max(candidates[name][loss], key=lambda row: row["bound"])
                   for loss in LOSSES} for name in shifts}
    available = {loss: {
        "route": max(AVAILABLE_SHIFTS, key=lambda name: best[name][loss]["bound"]),
        "bound": max(best[name][loss]["bound"] for name in AVAILABLE_SHIFTS),
    } for loss in LOSSES}
    return {"design": design, "seed": seed, "n": n,
            "claim_tier": "synthetic_diagnostic",
            "available_family_alpha": 0.05,
            "oracle_is_separate_non_deployable_ceiling": True,
            "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
            "best_by_shift": best, "available_portfolio": available}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=37001)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    for design in range(1, 11):
        for seed in range(args.seed_start, args.seed_start + args.reps):
            row = run(design, seed)
            with args.out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps({"design": design, "seed": seed,
                              "Q90": {name: row["best_by_shift"][name]["Q90"]["bound"]
                                      for name in ("zero", "booked", "oracle_mu")}}),
                  flush=True)
