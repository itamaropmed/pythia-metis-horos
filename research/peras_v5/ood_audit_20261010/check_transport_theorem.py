"""Known-law falsification checks for the conditional tuple theorem."""
import json

import numpy as np

from transport_tuple_bound import lower_bound


def exact_replicates(seed: int, n: int = 400, k: int = 8):
    rng = np.random.default_rng(seed)
    y = rng.choice((-10.0, 10.0), size=n)
    # Tuple design is fixed before outcomes; a cycle yields controlled overlap.
    tuples = (np.arange(n)[:, None] + np.arange(k)[None, :]) % n
    zero = np.zeros(tuples.shape)
    gold = {"MAE": 10.0, "Q90": 1.0, "MSE": 100.0}
    out = {}
    for loss in gold:
        b = lower_bound(y, tuples, zero, loss=loss, clip=(-10.0, 10.0),
                        alpha=0.05 / 3, cap_provenance="exact identical conditional laws")
        out[loss] = {"bound": b["bound"], "violates": b["bound"] > gold[loss] + 1e-10,
                     "radius": b["concentration_radius"]}
    return out


if __name__ == "__main__":
    runs = [exact_replicates(seed) for seed in range(500)]
    summary = {
        loss: {
            "violations": sum(row[loss]["violates"] for row in runs),
            "positive": sum(row[loss]["bound"] > 0 for row in runs),
            "median_bound": float(np.median([row[loss]["bound"] for row in runs])),
            "gold": {"MAE": 10.0, "Q90": 1.0, "MSE": 100.0}[loss],
        }
        for loss in ("MAE", "Q90", "MSE")
    }
    print(json.dumps(summary, indent=2))
