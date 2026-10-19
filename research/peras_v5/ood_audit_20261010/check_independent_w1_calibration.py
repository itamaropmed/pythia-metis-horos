"""Known-law coverage and refusal checks for an independently learned W1 cap."""
from __future__ import annotations

import json

import numpy as np

from disjoint_transport_bound import lower_bound
from independent_w1_calibration import (
    calibrate_group_w1, calibrate_mean_block_w1_direct, mean_ordered_block_cap,
)


def run(reps: int = 200) -> dict:
    probabilities = (0.50, 0.51)
    n_audit = 400
    n_cal_per_group = 20_000
    groups = np.tile((0, 1), n_audit // 2)
    blocks = np.arange(n_audit).reshape(n_audit // 2, 2)
    true_cross_w1 = 20 * abs(probabilities[1] - probabilities[0])
    true_ordered_mean_w1 = true_cross_w1 / 2
    gold = {
        "MAE": float(np.mean([20 * min(p, 1 - p) for p in probabilities])),
        "Q90": float(np.mean([2 * (1 - p) for p in probabilities])),
        "MSE": float(np.mean([400 * p * (1 - p) for p in probabilities])),
    }
    records = []
    for seed in range(reps):
        rng = np.random.default_rng(seed + 910_000)
        cal = {g: rng.choice((-10.0, 10.0), size=n_cal_per_group,
                             p=(1 - p, p)) for g, p in enumerate(probabilities)}
        fitted = calibrate_group_w1(cal, clip=(-10.0, 10.0), delta=0.025)
        cap = mean_ordered_block_cap(groups, blocks, fitted)
        direct = calibrate_mean_block_w1_direct(
            cal, groups, blocks, clip=(-10.0, 10.0), delta=0.025)
        direct_cap = direct["mean_w1_cap"]
        audit = np.array([10.0 if rng.random() < probabilities[g] else -10.0
                          for g in groups])
        row = {"seed": seed, "cap": cap,
               "cap_covers_true_w1": cap + 1e-12 >= true_ordered_mean_w1,
               "direct_cap": direct_cap,
               "direct_cap_covers_true_w1": direct_cap + 1e-12 >= true_ordered_mean_w1,
               "true_ordered_mean_w1": true_ordered_mean_w1}
        for loss, score_clip in (("MAE", 10.0), ("Q90", 2.0), ("MSE", 100.0)):
            out = lower_bound(audit, blocks, loss=loss, mean_w1_cap=cap,
                              score_clip=score_clip, alpha=0.025 / 3,
                              cap_provenance="independent group DKW calibration",
                              outcome_clip=(-10.0, 10.0))
            row[loss] = out["bound"]
            row[f"direct_{loss}"] = lower_bound(
                audit, blocks, loss=loss, mean_w1_cap=direct_cap,
                score_clip=score_clip, alpha=0.025 / 3,
                cap_provenance="independent direct average-W1 calibration",
                outcome_clip=(-10.0, 10.0))["bound"]
        records.append(row)
    assert all(row["cap_covers_true_w1"] for row in records)
    assert all(row["direct_cap_covers_true_w1"] for row in records)
    assert all(row[loss] <= gold[loss] + 1e-12
               for row in records for loss in gold)
    assert all(row[f"direct_{loss}"] <= gold[loss] + 1e-12
               for row in records for loss in gold)
    assert all(row["MAE"] > 0 for row in records)
    summary = {"reps": reps, "gold": gold,
               "true_mean_w1": true_ordered_mean_w1,
               "median_cap": float(np.median([r["cap"] for r in records])),
               "median_direct_cap": float(np.median([r["direct_cap"] for r in records])),
               "cap_failures": sum(not r["cap_covers_true_w1"] for r in records),
               "direct_cap_failures": sum(not r["direct_cap_covers_true_w1"] for r in records),
               "losses": {loss: {
                   "positive": sum(r[loss] > 0 for r in records),
                   "gold_exceedances": sum(r[loss] > gold[loss] + 1e-12 for r in records),
                   "median_bound": float(np.median([r[loss] for r in records])),
                   "direct_positive": sum(r[f"direct_{loss}"] > 0 for r in records),
                   "direct_gold_exceedances": sum(
                       r[f"direct_{loss}"] > gold[loss] + 1e-12 for r in records),
                   "direct_median_bound": float(np.median(
                       [r[f"direct_{loss}"] for r in records])),
               } for loss in gold}}
    # Refuse invalid calibration inputs and unknown group labels.
    for bad in ({0: np.array([0.]), 1: np.array([1., 2.])},
                {0: np.array([0., np.nan]), 1: np.array([1., 2.])}):
        try:
            calibrate_group_w1(bad, clip=(-10, 10))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid calibration sample must fail")
    try:
        mean_ordered_block_cap(np.full(n_audit, 3), blocks, fitted)
    except ValueError:
        pass
    else:
        raise AssertionError("uncalibrated group must fail")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
