"""Fresh known-law audit of the registered-support residual certificate."""
from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from generator_ood import make_ood
from pythia.core.residual_information import support_residual_information


def exact_nli(P, cells):
    value = 0.0
    for c in np.unique(cells):
        idx = cells == c
        value += ((P[idx] - P[idx].mean(0)) ** 2).sum()
    return float(value / len(cells))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rows = []
    for level, beta in (("null", 0.0), ("signal", 3.0)):
        for rep in range(reps):
            d = make_ood("O6_local_rare_token", n, beta, 880001 + 1009 * rep + int(beta * 10))
            result = support_residual_information(
                d["cell"], d["registered_support"], d["y"], 2,
                provenance={"support": "frozen O6 latent registry", "seed": rep},
            )
            rows.append({
                "level": level, "rep": rep, "truth": exact_nli(d["P"], d["cell"]),
                "lower": result["dI_lower"],
                "positive": result["dI_lower"] > 0,
                "valid": result["dI_lower"] <= exact_nli(d["P"], d["cell"]) + 1e-12,
                "resolution": result["resolution_status"],
            })
    print(json.dumps({
        "n": n, "reps_per_level": reps,
        "null_positive": sum(r["positive"] for r in rows if r["level"] == "null"),
        "signal_positive": sum(r["positive"] for r in rows if r["level"] == "signal"),
        "violations": sum(not r["valid"] for r in rows),
        "signal_mean_truth": float(np.mean([r["truth"] for r in rows if r["level"] == "signal"])),
        "signal_mean_lower": float(np.mean([r["lower"] for r in rows if r["level"] == "signal"])),
        "signal_median_fraction": float(np.median([
            r["lower"] / r["truth"] for r in rows
            if r["level"] == "signal" and r["truth"] > 0])),
    }, indent=2))
