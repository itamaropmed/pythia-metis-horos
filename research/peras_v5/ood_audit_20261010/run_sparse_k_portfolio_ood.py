"""Fresh selection-safe block-size portfolio for sparse fine representations."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from clipped_risk_gold import risks
from finite_group_multiloss_interval import (
    LOSSES,
    SCORE_CAPS,
    partial_group_lower,
    partial_group_portfolio,
)
from run_exact_law_battery import DESIGNS, make


BUCKETS = (5, 10, 20)
KS = (2, 4, 8, 16)
ALPHA_FAMILY = .05 / (len(LOSSES) * len(BUCKETS))


def run(design, seed):
    calibration = make(design, 20_000, seed + 2_000_000)
    audit = make(design, 800, seed)
    rng_cal = np.random.default_rng(seed + 8_300_000)
    rng_audit = np.random.default_rng(seed + 8_400_000)
    cal_u = rng_cal.integers(0, 2**31 - 1, len(calibration["y"]))
    audit_u = rng_audit.integers(0, 2**31 - 1, len(audit["y"]))
    base_groups = [(int(code), int(concept)) for code, concept
                   in zip(audit["code"], audit["concepts"][:, 0])]
    parameters = {}
    for i, group in enumerate(base_groups):
        current = (audit["fam"][i], audit["mu"][i], audit["s"][i])
        if group in parameters and parameters[group] != current:
            raise AssertionError("base group is not an exact law key")
        parameters[group] = current
    gold_by_group = {g: risks(*parameters[g]) for g in parameters}
    base_counts = Counter(base_groups)
    gold = {loss: sum((n / 800) * gold_by_group[g][loss]
                      for g, n in base_counts.items()) for loss in LOSSES}

    levels = []
    for buckets in BUCKETS:
        samples = defaultdict(list)
        for code, concept, u, y in zip(
                calibration["code"], calibration["concepts"][:, 0],
                cal_u, calibration["y"]):
            samples[(int(code), int(concept), int(u % buckets))].append(float(y))
        fine_groups = [(g[0], g[1], int(u % buckets))
                       for g, u in zip(base_groups, audit_u)]
        weights = {g: n / 800 for g, n in Counter(fine_groups).items()}
        losses = {}
        for loss in LOSSES:
            fixed = partial_group_lower(
                samples, weights, loss=loss, k=16, score_cap=SCORE_CAPS[loss],
                alpha_endpoint=ALPHA_FAMILY)
            portfolio = partial_group_portfolio(
                samples, weights, loss=loss, ks=KS,
                score_cap=SCORE_CAPS[loss], alpha_family=ALPHA_FAMILY)
            losses[loss] = {
                "gold": float(gold[loss]),
                "fixed_k16_lower": fixed["lower"],
                "fixed_k16_mass": fixed["included_target_mass"],
                "portfolio_lower": portfolio["lower"],
                "portfolio_mass": portfolio["included_target_mass"],
                "selected_k": portfolio["selected_k"],
            }
        levels.append({"n_nuisance_buckets": buckets, "losses": losses})
    return {"design": design, "seed": seed, "n_audit": 800,
            "n_calibration": 20_000, "candidate_ks": KS,
            "alpha_family_per_loss_bucket": ALPHA_FAMILY,
            "levels": levels}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=100001)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists")
    with args.out.open("w") as handle:
        for design in DESIGNS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(design, seed, flush=True)
