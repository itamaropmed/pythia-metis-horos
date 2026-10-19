"""Known-law checks of disjoint score-clipped transport implementation."""
from __future__ import annotations

import json

import numpy as np

from disjoint_transport_bound import lower_bound


def run(reps: int = 500) -> dict:
    blocks = np.arange(400).reshape(50, 8)
    truth = {"MAE": 10.0, "Q90": 1.0, "MSE": 100.0}
    clips = {"MAE": 10.0, "Q90": 2.0, "MSE": 100.0}
    results = {loss: [] for loss in truth}
    for seed in range(reps):
        y = np.random.default_rng(seed).choice((-10.0, 10.0), size=400)
        for loss in truth:
            result = lower_bound(
                y, blocks, loss=loss, mean_w1_cap=0.0,
                score_clip=clips[loss], alpha=0.05 / 3,
                cap_provenance="known exact Bernoulli law",
                outcome_clip=(-10.0, 10.0) if loss == "MSE" else None,
            )
            assert result["status"] == "conditional"
            assert 0 <= result["bound"] <= result["lower_score_mean"] + 1e-12
            results[loss].append(result["bound"])
    summary = {loss: {
        "gold": value,
        "positive": sum(bound > 0 for bound in results[loss]),
        "violations": sum(bound > value + 1e-12 for bound in results[loss]),
        "median_bound": float(np.median(results[loss])),
    } for loss, value in truth.items()}
    try:
        lower_bound(np.arange(6.0), np.array([[0, 1], [1, 2]]),
                    loss="MAE", mean_w1_cap=0.0, score_clip=1.0,
                    cap_provenance="invalid overlapping blocks")
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping blocks must fail")
    assert lower_bound(np.arange(6.0), np.array([[0, 1], [2, 3]]),
                       loss="W30", mean_w1_cap=0.0, score_clip=1.0,
                       cap_provenance="unsupported loss")["status"] == "refused"
    # Nonconstant, predeclared shifts align exact residual laws and preserve
    # every translation-invariant Bayes-risk target.
    y0 = np.random.default_rng(105).choice((-10.0, 10.0), size=400)
    shift = np.linspace(-100.0, 100.0, 400)
    for loss in truth:
        kwargs = {"loss": loss, "mean_w1_cap": 0.0,
                  "score_clip": clips[loss], "alpha": 0.05 / 3,
                  "cap_provenance": "known exact residual law",
                  "outcome_clip": (-10.0, 10.0) if loss == "MSE" else None}
        baseline = lower_bound(y0, blocks, **kwargs)
        aligned = lower_bound(y0 + shift, blocks, shift=shift,
                              shift_provenance="frozen synthetic pre-outcome shift", **kwargs)
        assert np.isclose(baseline["bound"], aligned["bound"], atol=1e-12)
    # A false zero-W1 assertion can manufacture a positive lower bound even
    # when every case has a deterministic outcome and Bayes risk is exactly 0.
    deterministic_y = np.tile((-10.0, 10.0), 200)
    false = lower_bound(deterministic_y, blocks, loss="MAE",
                        mean_w1_cap=0.0, score_clip=10.0,
                        cap_provenance="deliberately false zero cap")
    correct = lower_bound(deterministic_y, blocks, loss="MAE",
                          mean_w1_cap=10.0, score_clip=10.0,
                          cap_provenance="exact ordered within-block mean W1")
    assert false["bound"] > 0 and correct["bound"] == 0
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
