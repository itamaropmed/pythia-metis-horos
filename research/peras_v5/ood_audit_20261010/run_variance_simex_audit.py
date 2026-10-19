"""Compare frozen linear SIMEX with a Gaussian-motivated square transform.

The OOD mechanisms and gold integration are frozen in audit_ood.py. This
script does not train a predictor and does not construct a confidence bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import audit_ood
import gold
import peras5 as P
import simex_variance_stabilized as S


def run(design: int, seed: int, n: int = 800) -> dict:
    d = audit_ood.make(design, n, seed)
    g = gold.pointwise(d["fam"], d["s"])
    est = S.estimate(d, seed=seed)
    truth = {loss: float(np.mean(g[loss])) for loss in P.LOSSES}
    return dict(design=design, name=audit_ood.NAMES[design], seed=seed,
                n=n, gold=truth, linear=est["linear"],
                squared=est["squared"],
                selected_columns=est["selected_columns"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-seed", type=int, required=True)
    ap.add_argument("--last-seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.first_seed > args.last_seed:
        ap.error("first seed must not exceed last seed")
    # A separate direct call guards against implementation drift in the
    # frozen-linear branch, which is intentionally copied from peras54.
    parity_data = audit_ood.make(1, 800, args.first_seed)
    paired = S.estimate(parity_data, seed=args.first_seed)["linear"]
    frozen = P.peras57(parity_data, seed=args.first_seed)[0]
    for loss in frozen:
        if not np.isclose(paired[loss], frozen[loss], atol=1e-10, rtol=1e-10):
            raise AssertionError(f"frozen parity failed for {loss}")
    with args.output.open("w") as f:
        for seed in range(args.first_seed, args.last_seed + 1):
            for design in range(1, 11):
                row = run(design, seed)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(seed, design, row["name"], flush=True)


if __name__ == "__main__":
    main()
