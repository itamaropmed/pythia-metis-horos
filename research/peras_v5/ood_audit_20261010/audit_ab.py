"""Matched n=200 comparison to the published Anderson--Bjarnadottir protocol."""
import argparse
import json
from pathlib import Path

import numpy as np

from omniscient_lp_baseline import anderson_bjarnadottir_mae_lp

from audit_ood import make
import gold
import peras5 as P


ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path)
args = ap.parse_args()
if args.out is not None and args.out.exists():
    ap.error(f"output already exists: {args.out}; choose a new file")
for design in range(1, 11):
    for seed in (27001, 27002, 27003):
        d = make(design, 200, seed)
        R = np.column_stack([d["tab"], d["llm"]])
        a = anderson_bjarnadottir_mae_lp(R, d["y"])
        e, b, _ = P.peras57(d, seed=seed)
        g = float(gold.pointwise(d["fam"], d["s"])["MAE"].mean())
        row = dict(design=design, seed=seed, gold=g,
                   peras=e["MAE"], bound=b["MAE"],
                   ab=a.get("empirical_omniscient_mae"), ab_status=a["status"])
        if args.out is not None:
            with args.out.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
