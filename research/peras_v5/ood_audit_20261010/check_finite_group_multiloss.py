"""Analytic two-point gold and refusal tests for eight group-risk losses."""
import json

import numpy as np

from finite_group_multiloss_interval import (
    LOSSES, SCORE_CAPS, balanced_bentkus_interval, interval,
    randomized_mixture_bentkus_interval,
)
from refinement_gap_profile import transfer_bound


def run(reps=200):
    p = np.array([.5, .51])
    gold = {"MAE": float(np.mean(80 * np.minimum(p, 1 - p))),
            "MSE": float(np.mean(6400 * p * (1 - p))),
            "Q10": float(np.mean(8 * p)),
            "Q50": float(np.mean(40 * np.minimum(p, 1 - p))),
            "Q90": float(np.mean(8 * (1 - p))),
            "W15": float(np.mean(np.minimum(p, 1 - p))),
            "W30": float(np.mean(np.minimum(p, 1 - p))), "W60": 0.}
    counts = {loss: 0 for loss in LOSSES}
    for seed in range(reps):
        rng = np.random.default_rng(seed + 720_000)
        samples = {g: rng.choice((-40., 40.), 800, p=(1 - prob, prob))
                   for g, prob in enumerate(p)}
        for loss in LOSSES:
            r = interval(samples, {0: .5, 1: .5}, loss=loss, clip=(-40., 40.),
                         score_cap=SCORE_CAPS[loss])
            bentkus = balanced_bentkus_interval(
                samples, {0: 20, 1: 20}, loss=loss, clip=(-40., 40.),
                score_cap=SCORE_CAPS[loss])
            randomized = randomized_mixture_bentkus_interval(
                samples, {0: 20, 1: 20}, loss=loss,
                randomization_seed=880_000 + seed, n_blocks=40,
                clip=(-40., 40.), score_cap=SCORE_CAPS[loss])
            assert r["status"] == "conditional"
            assert bentkus["status"] == "conditional"
            assert randomized["status"] == "conditional"
            assert 0 <= r["lower"] <= r["upper"]
            assert 0 <= bentkus["lower"] <= bentkus["upper"]
            assert 0 <= randomized["lower"] <= randomized["upper"]
            assert transfer_bound(r, bound_key="lower")["status"] == "refused"
            disclosed = transfer_bound(r, bound_key="lower",
                                       risk_gap_cap=gold[loss],
                                       cap_provenance="known full-disclosure synthetic DGP")
            assert disclosed["bound"] == 0
            if loss.startswith("W"):
                assert transfer_bound(r, bound_key="lower", mean_w1_cap=1.,
                                      cap_provenance="unsupported W1 route")["status"] == "refused"
            counts[loss] += not r["lower"] - 1e-12 <= gold[loss] <= r["upper"] + 1e-12
            counts.setdefault(f"bentkus_{loss}", 0)
            counts[f"bentkus_{loss}"] += not (
                bentkus["lower"] - 1e-12 <= gold[loss] <= bentkus["upper"] + 1e-12)
            counts.setdefault(f"randomized_{loss}", 0)
            counts[f"randomized_{loss}"] += not (
                randomized["lower"] - 1e-12 <= gold[loss] <= randomized["upper"] + 1e-12)
    assert all(n == 0 for n in counts.values())
    assert interval({0: [1.]}, {0: 1.}, loss="MAE")["status"] == "refused"
    assert interval(samples, {2: 1.}, loss="MAE")["status"] == "refused"
    assert interval(samples, {0: 1.}, loss="unsupported")["status"] == "refused"
    assert balanced_bentkus_interval(
        {0: np.arange(16.)}, {0: 2}, loss="MAE")["status"] == "refused"
    out = {"reps": reps, "loss_cases": reps * len(LOSSES),
           "gold": gold, "interval_misses": counts}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
