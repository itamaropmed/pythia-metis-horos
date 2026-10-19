"""Frozen fresh OOD audit: barycentric vs pairwise coupled-environment penalties, with and without a
sample-split environment-median alignment.  PRE-REGISTERED before any run: seeds 103001-103005, designs 1-10,
n = 800, 5,000 environments per coupling, tuple sizes (2,4,8), the existing score clips, outcome clip [0,300]
(aligned routes: residual clip [-150,150]), cap budget .025 split over tuple sizes (and, for barycentric, over the
two penalty families and concentration routes), risk budget .025 split over 3 k x 3 losses x 3 score clips.
Each route is a separate 95% procedure; no post-hoc maximum over routes is reported as a certificate."""
import argparse, json
from pathlib import Path
import numpy as np
from audit_ood import make
import gold, peras5 as P
from coupled_environment_w1 import calibrate_coupled_mean_w1
from disjoint_transport_bound import lower_bound
from barycentric_coupled_bound import calibrate_bary, lower_bound_bary
from run_coupled_environment_ood import KS, LOSSES, SCORE_CLIPS, OUTCOME_CLIP, CAP_DELTA_FAMILY, BOUND_ALPHA, repeated_outcomes
from run_disjoint_transport_audit import greedy_feature_blocks
RESIDUAL_CLIP = (-150., 150.)


def pairwise(d, env, blocks, prov, clip, shift=None):
    out = {L: [] for L in LOSSES}
    for k, b in blocks.items():
        cap = calibrate_coupled_mean_w1(env, b, clip=clip, delta_family=CAP_DELTA_FAMILY, coupling_provenance=prov,
                                        shift=shift, shift_provenance="env-median split" if shift is not None else None)
        for L in LOSSES:
            for sc in SCORE_CLIPS[L]:
                r = lower_bound(d["y"], b, loss=L, mean_w1_cap=cap["mean_w1_cap"], score_clip=sc, alpha=BOUND_ALPHA,
                                cap_provenance=prov, outcome_clip=clip, shift=shift,
                                shift_provenance="env-median split" if shift is not None else None)
                out[L].append((r["bound"], k))
    return {L: max(v) for L, v in out.items()}


def bary(d, env, blocks, prov, clip, shift=None):
    out = {L: [] for L in LOSSES}
    for k, b in blocks.items():
        cap = calibrate_bary(env, b, clip=clip, delta_family=CAP_DELTA_FAMILY, coupling_provenance=prov,
                             shift=shift, shift_provenance="env-median split" if shift is not None else None)
        for L in LOSSES:
            for sc in SCORE_CLIPS[L]:
                r = lower_bound_bary(d["y"], b, loss=L, cap=cap, score_clip=sc, alpha=BOUND_ALPHA, outcome_clip=clip, shift=shift)
                out[L].append((r["bound"], k))
    return {L: max(v) for L, v in out.items()}


def run(design, seed, n=800, E=5000):
    d = make(design, n, seed); rep = P.represent(d)
    blocks = {k: greedy_feature_blocks(rep, k) for k in KS}
    g = gold.pointwise(d["fam"], d["s"]); row = {"design": design, "seed": seed, "gold": {L: float(np.mean(g[L])) for L in LOSSES}}
    for cname, common in [("common", True), ("product", False)]:
        env = repeated_outcomes(d, E, seed + (9_400_000 if common else 9_500_000), common_quantile=common)
        prov = f"registered {cname} synthetic environments"
        row[f"pair_{cname}"] = pairwise(d, env, blocks, prov, OUTCOME_CLIP)
        row[f"bary_{cname}"] = bary(d, env, blocks, prov, OUTCOME_CLIP)
        learn, cal = np.array_split(env, 2, axis=0); shift = np.median(learn, axis=0)
        row[f"pairAL_{cname}"] = pairwise(d, cal, blocks, prov, RESIDUAL_CLIP, shift)
        row[f"baryAL_{cname}"] = bary(d, cal, blocks, prov, RESIDUAL_CLIP, shift)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=103001); ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.out.exists(): ap.error("output exists")
    with a.out.open("w") as h:
        for design in range(1, 11):
            for seed in range(a.first_seed, a.first_seed + a.reps):
                r = run(design, seed); h.write(json.dumps(r, default=float) + "\n"); h.flush()
                print(design, seed, {k: {L: round(v[L][0], 2) for L in LOSSES} for k, v in r.items() if isinstance(v, dict) and k != "gold"}, flush=True)
