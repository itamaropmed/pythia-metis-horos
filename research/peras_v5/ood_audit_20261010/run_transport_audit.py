"""Test a proved conditional tuple bound with oracle W1 caps on the OOD suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_ood import make
import gold
import peras5 as P
from transport_tuple_bound import lower_bound


# Exact E[Z^2] for noise families in gold.py.  Cauchy-Schwarz makes these
# conservative upper bounds on E|Z|, so the resulting W1 caps are valid.
SECOND_MOMENT = {
    "normal": 1.0, "laplace": 2.0, "gamma2": 6.0,
    "lognorm": float(np.exp(0.72)), "t3": 3.0, "t5": 5.0 / 3.0,
    "mix05": 1.0 + 12.25 * 0.05,
    "mix35": 1.0 + 12.25 * 0.35,
}


def oracle_transport_caps(d: dict, tuples: np.ndarray) -> np.ndarray:
    """Analytic upper cap using a shared-noise or independent-noise coupling."""
    mu = d["mu"]
    scale = d["s"]
    fam = d["fam"]
    anchor = tuples[:, :1]
    base = np.abs(mu[anchor] - mu[tuples])
    ea = np.array([np.sqrt(SECOND_MOMENT[f]) for f in fam[anchor].ravel()])[:, None]
    eb = np.array([np.sqrt(SECOND_MOMENT[f]) for f in fam[tuples].ravel()]).reshape(tuples.shape)
    same = fam[anchor] == fam[tuples]
    cap_same = base + np.abs(scale[anchor] - scale[tuples]) * ea
    cap_other = base + scale[anchor] * ea + scale[tuples] * eb
    cap = np.where(same, cap_same, cap_other)
    cap[:, 0] = 0.0
    return cap


def run(design: int, seed: int, n: int = 800, k: int = 8) -> dict:
    d = make(design, n, seed)
    Z = P.represent(d)
    _, nb = P.neighbours(Z, k - 1)
    tuples = np.column_stack([np.arange(n), nb])
    caps = oracle_transport_caps(d, tuples)
    g = gold.pointwise(d["fam"], d["s"])
    results = {}
    for loss in ("MAE", "Q90", "MSE"):
        results[loss] = lower_bound(
            d["y"], tuples, caps, loss=loss, clip=(40.0, 280.0),
            alpha=0.05 / 3.0,
            cap_provenance="oracle analytic coupling from synthetic DGP, not deployable",
        )
    return {
        "design": design, "seed": seed, "gold": {loss: float(g[loss].mean()) for loss in results},
        "bounds": results, "n": n, "k": k,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"output already exists: {args.out}; choose a new file")
    for design in range(1, 11):
        for seed in (27001, 27002, 27003):
            row = run(design, seed)
            with args.out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"design": design, "seed": seed,
                              "MAE_bound": row["bounds"]["MAE"]["bound"],
                              "MAE_gold": row["gold"]["MAE"]}), flush=True)
