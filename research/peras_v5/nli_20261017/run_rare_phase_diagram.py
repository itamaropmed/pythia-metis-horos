"""Known-law rare-signal phase diagram: generic matching vs support witness."""
from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)

from pythia.core.anchored_information import anchored_information
from pythia.core.residual_information import support_residual_information


def one(prevalence, beta, n, seed):
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 8, n)
    rare = rng.random(n) < prevalence
    base = .28 * (cells - 3.5)
    p = 1 / (1 + np.exp(-(base + beta * rare)))
    y = (rng.random(n) < p).astype(int)
    P = np.column_stack([1 - p, p])
    truth = 0.0
    for c in np.unique(cells):
        idx = cells == c
        truth += ((P[idx] - P[idx].mean(0)) ** 2).sum()
    truth /= n
    generic = anchored_information(cells, rare[:, None].astype(float), y, 2, alpha=.05, seed=seed)
    support = support_residual_information(
        cells, rare.astype(int), y, 2, alpha=.05,
        provenance={"registry": "predeclared rare concept", "seed": seed})
    return truth, generic["dI_lower"], support["dI_lower"]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rows = []
    design = 0
    for prevalence in (.01, .02, .05, .10, .20):
        for beta in (1.5, 3.0):
            runs = [one(prevalence, beta, n, 660001 + 10007 * design + r)
                    for r in range(reps)]
            truth = np.array([r[0] for r in runs])
            generic = np.array([r[1] for r in runs])
            support = np.array([r[2] for r in runs])
            rows.append({
                "design": design, "prevalence": prevalence, "beta": beta,
                "mean_truth": float(truth.mean()),
                "generic_positive": int(np.sum(generic > 0)),
                "support_positive": int(np.sum(support > 0)),
                "generic_median_fraction": float(np.median(generic / truth)),
                "support_median_fraction": float(np.median(support / truth)),
                "support_strict_wins": int(np.sum(support > generic + 1e-12)),
                "support_violations": int(np.sum(support > truth + 1e-12)),
            })
            design += 1
    print(json.dumps({"n": n, "reps": reps, "designs": rows}, indent=2))
