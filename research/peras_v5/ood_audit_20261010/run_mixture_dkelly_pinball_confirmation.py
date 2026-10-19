"""Untouched confirmation of the frozen Q90 partition-mixture e-value."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from peras import mixture_dkelly_pinball_portfolio
from peras.core.concentration import bentkus_lower_mean
from peras.core.truncated_saa import dkelly_lower_mean
from run_multipoint_pinball_development import (
    Q, finite_range, gold, partitions, quantile_function, score,
)


K = 16
CAP = .08


def run(design, seed):
    y = quantile_function(design, np.random.default_rng(seed).random(800))
    registered = partitions(800, seed)[K]
    mixture = mixture_dkelly_pinball_portfolio(
        y, {K: registered}, quantile=Q, clip=(0, 1),
        declared_partition_discrepancies={K: np.zeros(len(registered))},
        score_caps=(CAP,), alpha_family=.05,
        partition_provenance="20 frozen random balanced partitions",
        discrepancy_provenance="iid exact-law mechanism",
        portfolio_frozen_before_outcomes=True)
    first_scores = score(y, registered[0])
    single = dkelly_lower_mean(
        np.minimum(first_scores, CAP), cap=CAP, alpha=.05)
    pairs = registered[0].reshape(-1)[:len(y)].reshape(-1, 2)
    pair_scores = score(y, pairs)
    paired = bentkus_lower_mean(pair_scores, finite_range(2), .05)
    return {"design": design, "seed": seed, "gold": gold(design),
            "mixture_dkelly": mixture.bound,
            "single_partition_dkelly": float(single),
            "paired_bentkus": float(paired)}


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1,11):
            for seed in range(137001,137101):
                handle.write(json.dumps(run(design,seed))+"\n");handle.flush()
        print(args.out)
