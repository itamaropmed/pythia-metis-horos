"""Known-floor and numerical-invariance checks for grouped SAA references."""
from __future__ import annotations

import json

import numpy as np

from group_calibration_floor import grouped_floor


def run(reps: int = 200) -> dict:
    truth = {"MAE": 9.9, "Q90": 0.99, "MSE": 99.98}
    clips = {"MAE": 10., "Q90": 2., "MSE": 100.}
    counts = {loss: {"tuple_exceedances": 0, "full_exceedances": 0}
              for loss in truth}
    for seed in range(reps):
        rng = np.random.default_rng(seed + 660_000)
        samples = {0: rng.choice((-10., 10.), 800, p=(.5, .5)),
                   1: rng.choice((-10., 10.), 800, p=(.49, .51))}
        for loss in truth:
            out = grouped_floor(samples, {0: .5, 1: .5}, loss=loss,
                                clip=(-10., 10.), score_clip=clips[loss])
            assert 0 <= out["tuple_bound"] <= out["tuple_center"]
            assert 0 <= out["full_saa_bound"] <= out["plugin_clipped"]
            counts[loss]["tuple_exceedances"] += out["tuple_bound"] > truth[loss] + 1e-12
            counts[loss]["full_exceedances"] += out["full_saa_bound"] > truth[loss] + 1e-12
            shifted = grouped_floor({g: y + 37. for g, y in samples.items()},
                                    {0: .5, 1: .5}, loss=loss,
                                    clip=(27., 47.), score_clip=clips[loss])
            for key in ("tuple_bound", "full_saa_bound", "plugin_original"):
                assert np.isclose(out[key], shifted[key], atol=1e-10)
    for weights in ({0: .5}, {0: 1.1, 1: -.1}):
        try:
            grouped_floor(samples, weights, loss="MAE", clip=(-10, 10), score_clip=10)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid weights must fail")
    print(json.dumps({"reps": reps, "gold": truth, "results": counts}, indent=2))
    return counts


if __name__ == "__main__":
    run()
