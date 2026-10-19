"""Fresh OOD test of disjoint, score-clipped near-replicate tuple bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from audit_ood import make
from disjoint_transport_bound import lower_bound
import gold
import peras5 as P
from run_transport_audit import oracle_transport_caps


KS = (2, 4, 8)
SCORE_CLIPS = {"MAE": (10.0, 20.0, 40.0),
               "Q90": (2.0, 5.0, 10.0),
               "MSE": (200.0, 400.0, 800.0)}
LOSSES = ("MAE", "Q90", "MSE")
ALPHA = 0.05 / (len(KS) * len(LOSSES) * len(SCORE_CLIPS["MAE"]))
MSE_OUTCOME_CLIP = (40.0, 280.0)


def greedy_feature_blocks(Z: np.ndarray, k: int) -> np.ndarray:
    """Deterministic, outcome-blind grouping with globally disjoint entities."""
    Z = np.asarray(Z, float)
    if Z.ndim != 2 or len(Z) < 2 * k or not np.isfinite(Z).all():
        raise ValueError("finite feature matrix with >=2k rows required")
    dist = cdist(Z, Z)
    np.fill_diagonal(dist, np.inf)
    density_radius = np.partition(dist, k - 1, axis=1)[:, k - 1]
    order = np.argsort(density_radius, kind="stable")
    available = np.ones(len(Z), dtype=bool)
    blocks = []
    for anchor in order:
        if not available[anchor] or np.sum(available) < k:
            continue
        choices = np.flatnonzero(available)
        choices = choices[choices != anchor]
        selected = choices[np.argsort(dist[anchor, choices], kind="stable")[:k - 1]]
        block = np.r_[anchor, selected]
        blocks.append(block)
        available[block] = False
    result = np.asarray(blocks, dtype=int)
    if result.ndim != 2 or result.shape[1] != k or np.unique(result).size != result.size:
        raise AssertionError("feature matching failed to produce disjoint blocks")
    return result


def oracle_mean_within_block_w1(d: dict, blocks: np.ndarray) -> float:
    """Conservative mean ordered W1 cap from known location-scale generator."""
    k = blocks.shape[1]
    values = []
    for anchor_position in range(k):
        others = [j for j in range(k) if j != anchor_position]
        reordered = np.column_stack([blocks[:, anchor_position], blocks[:, others]])
        values.append(float(np.mean(oracle_transport_caps(d, reordered))))
    return float(np.mean(values))


def run(design: int, seed: int, n: int = 800) -> dict:
    d = make(design, n, seed)
    Z = P.represent(d)
    truth = gold.pointwise(d["fam"], d["s"])
    candidates = {loss: [] for loss in LOSSES}
    for k in KS:
        blocks = greedy_feature_blocks(Z, k)
        cap = oracle_mean_within_block_w1(d, blocks)
        for loss in LOSSES:
            for score_clip in SCORE_CLIPS[loss]:
                b = lower_bound(
                    d["y"], blocks, loss=loss, mean_w1_cap=cap,
                    score_clip=score_clip, alpha=ALPHA,
                    cap_provenance="analytic oracle coupling from synthetic DGP",
                    outcome_clip=MSE_OUTCOME_CLIP if loss == "MSE" else None,
                )
                candidates[loss].append(b)
    return {"design": design, "seed": seed, "n": n,
            "claim_tier": "synthetic_diagnostic",
            "gold": {loss: float(truth[loss].mean()) for loss in LOSSES},
            "chosen": {loss: max(candidates[loss], key=lambda row: row["bound"])
                       for loss in LOSSES},
            "candidate_counts": {loss: {
                "positive": sum(row["bound"] > 0 for row in candidates[loss]),
                "score_limited": sum(row["lower_score_mean"] <= 0 for row in candidates[loss]),
                "transport_limited": sum(row["lower_score_mean"] > 0 and row["bound"] == 0
                                         for row in candidates[loss]),
            } for loss in LOSSES}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=36001)
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
                              "bounds": {loss: row["chosen"][loss]["bound"] for loss in LOSSES}}),
                  flush=True)
