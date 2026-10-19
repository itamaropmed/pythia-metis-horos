"""Untouched confirmation of the frozen repeated-modulus V1 loss router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from peras import calibrate_repeated_environment_discrepancy_v1
from run_repeated_modulus_calibration_ood import mechanism, true_costs


def run(design: int, seed: int):
    location, scale, features, blocks = mechanism(design)
    u = np.random.default_rng(seed).random((500, 1))
    replicates = location + scale*(u-.5)
    truth_abs, truth_sq = true_costs(location, scale, blocks)
    result = calibrate_repeated_environment_discrepancy_v1(
        replicates, blocks, features, clip=(-5, 5), alpha_family=.05,
        marginal_contract="target_unit_conditional",
        coupling_provenance="registered common-U target coupling",
        metric_provenance="registered Euclidean [x,note] metric",
        portfolio_frozen_before_replicates=True)
    return {
        "design": design, "seed": seed,
        "true_absolute": truth_abs, "true_squared": truth_sq,
        "absolute_cap": result.absolute_discrepancy_cap,
        "squared_cap": result.squared_discrepancy_cap,
        "absolute_candidates": result.candidates["absolute"],
        "squared_candidates": result.candidates["squared"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1, 11):
            for seed in range(134001, 134101):
                handle.write(json.dumps(run(design, seed)) + "\n")
                handle.flush()
        print(args.out)
