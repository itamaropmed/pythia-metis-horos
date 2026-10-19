"""Outcome-blind tuple portfolio: tolerated mean W1 versus oracle upper cap.

The caps here use the known synthetic DGP, so the results falsify or diagnose
the theorem's practical nonvacuity. They do not provide deployable cap
provenance or certify Mayo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
import gold
import peras5 as P
from run_transport_audit import oracle_transport_caps
from run_transport_portfolio import ALPHA, CLIPS, KS, LOSSES
from transport_tuple_bound import average_w1_sensitivity


def run(design: int, seed: int, n: int = 800) -> dict:
    d = make(design, n, seed)
    Z = P.represent(d)
    _, nb = P.neighbours(Z, max(KS) - 1)
    truth = gold.pointwise(d["fam"], d["s"])
    candidates = {loss: [] for loss in LOSSES}
    for k in KS:
        idx = np.column_stack([np.arange(n), nb[:, :k - 1]])
        oracle_caps = oracle_transport_caps(d, idx)
        mean_oracle_cap = float(np.mean(oracle_caps))
        for clip in CLIPS:
            for loss in LOSSES:
                profile = average_w1_sensitivity(
                    d["y"], idx, loss=loss, clip=clip,
                    mean_w1_grid=np.asarray([mean_oracle_cap]), alpha=ALPHA,
                )
                if profile["status"] != "sensitivity_only":
                    raise AssertionError("unexpected refused loss")
                intercept = profile["statistic"] - profile["concentration_radius"]
                candidates[loss].append({
                    "bound_at_oracle_cap": profile["bounds"][0],
                    "tolerated_mean_w1": profile["breakpoint"],
                    "mean_oracle_upper_cap": mean_oracle_cap,
                    "concentration_intercept": intercept,
                    "concentration_limited": intercept <= 0,
                    "transport_limited": intercept > 0 and mean_oracle_cap >= profile["breakpoint"],
                    "tuple_size": k, "clip": list(clip),
                })
    return {"design": design, "seed": seed, "n": n,
            "oracle_cap_source": "analytic upper coupling using known synthetic location-scale DGP",
            "claim_tier": "synthetic_diagnostic",
            "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
            "chosen": {loss: max(candidates[loss], key=lambda r: r["bound_at_oracle_cap"])
                       for loss in LOSSES},
            "max_tolerated_mean_w1": {
                loss: max(r["tolerated_mean_w1"] for r in candidates[loss])
                for loss in LOSSES
            },
            "candidate_counts": {
                loss: {"concentration_limited": sum(r["concentration_limited"] for r in candidates[loss]),
                       "transport_limited": sum(r["transport_limited"] for r in candidates[loss]),
                       "positive": sum(r["bound_at_oracle_cap"] > 0 for r in candidates[loss])}
                for loss in LOSSES
            }}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=35001)
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
                              "MAE": row["chosen"]["MAE"]}), flush=True)
