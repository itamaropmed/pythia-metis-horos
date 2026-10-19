"""Deterministic contraction and known-tail coverage checks."""
import json

import numpy as np

from finite_group_multiloss_interval import empirical_floor_all, interval
from tail_transfer import penalty_values, tail_upper, transfer_interval


def run(reps=500):
    losses = ("MAE", "MSE", "Q10", "Q50", "Q90", "W15", "W30", "W60")
    contraction_violations = 0
    window_misses = {loss: 0 for loss in losses if loss.startswith("W")}
    for seed in range(reps):
        rng = np.random.default_rng(seed + 810_000)
        y = rng.standard_t(3, 800) * 80 + 130
        clipped = np.clip(y, 0, 300)
        for loss in losses:
            original = empirical_floor_all(y, loss)
            clipped_risk = empirical_floor_all(clipped, loss)
            penalty = float(penalty_values(y, loss=loss, clip=(0, 300)).mean())
            contraction_violations += not (
                clipped_risk <= original + 1e-10
                and original <= clipped_risk + penalty + 1e-10)
        # Independent calibration/target draws from a fixed two-group law.
        calibration = {0: rng.standard_t(3, 800) * 80 + 130,
                       1: rng.standard_t(3, 800) * 60 + 160}
        target = {0: rng.standard_t(3, 200_000) * 80 + 130,
                  1: rng.standard_t(3, 200_000) * 60 + 160}
        for loss in window_misses:
            tail = tail_upper(calibration, {0: .5, 1: .5}, loss=loss,
                              clip=(0, 300), alpha=.025)
            truth = .5 * np.mean((target[0] < 0) | (target[0] > 300))
            truth += .5 * np.mean((target[1] < 0) | (target[1] > 300))
            window_misses[loss] += tail["tail_penalty_upper"] + 1e-12 < truth
    assert contraction_violations == 0
    # Monte Carlo target error makes this diagnostic approximate; the observed
    # miss count is reported rather than used as proof of Hoeffding coverage.
    dummy = interval({0: np.arange(32.)}, {0: 1.}, loss="MAE")
    assert tail_upper({0: np.arange(32.)}, {0: 1.}, loss="MAE",
                      clip=(0, 10), alpha=.05)["status"] == "refused"
    assert transfer_interval(dummy, {"status": "refused"})["status"] == "refused"
    out = {"reps": reps, "deterministic_loss_checks": reps * len(losses),
           "contraction_violations": contraction_violations,
            "window_tail_cap_misses_against_large_MC_target": {
                key: int(value) for key, value in window_misses.items()}}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
