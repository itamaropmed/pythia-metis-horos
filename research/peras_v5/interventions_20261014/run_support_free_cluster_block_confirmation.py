#!/usr/bin/env python3
"""Frozen run for SUPPORT_FREE_CLUSTER_BLOCK_REGISTRATION.md."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gamma

from peras import clustered_block_saa_portfolio, truncated_saa_portfolio


SQRT_2_PI = math.sqrt(2 / math.pi)


def _normal_absolute_mean(mu, sigma):
    return (sigma * SQRT_2_PI * math.exp(-mu * mu / (2 * sigma * sigma))
            + abs(mu) * math.erf(abs(mu) / (sigma * math.sqrt(2))))


LAWS = (
    ("gaussian", lambda rng, n: rng.normal(size=n), SQRT_2_PI),
    ("laplace", lambda rng, n: rng.laplace(size=n), 1.),
    ("logistic", lambda rng, n: rng.logistic(size=n), 2 * math.log(2)),
    ("student_t3", lambda rng, n: rng.standard_t(3, size=n),
     2 * math.sqrt(3) / math.pi),
    ("student_t5", lambda rng, n: rng.standard_t(5, size=n),
     2 * math.sqrt(5) * gamma(3) / (4 * math.sqrt(math.pi) * gamma(2.5))),
    ("gaussian_scale_mixture",
     lambda rng, n: rng.normal(size=n) * rng.choice((1., 5.), n, p=(.8, .2)),
     1.8 * SQRT_2_PI),
    ("gaussian_location_mixture",
     lambda rng, n: rng.normal(rng.choice((-2., 2.), n), .5),
     _normal_absolute_mean(2., .5)),
    ("signed_lognormal",
     lambda rng, n: rng.choice((-1., 1.), n) * rng.lognormal(0., .5, n),
     math.exp(.125)),
    ("signed_exponential",
     lambda rng, n: rng.choice((-1., 1.), n) * rng.exponential(1., n), 1.),
    ("contaminated_gaussian",
     lambda rng, n: rng.normal(size=n) * rng.choice((1., 10.), n, p=(.95, .05)),
     1.45 * SQRT_2_PI),
)

CAPS = (.25, .5, 1., 2., 4., 8.)
BLOCK_SIZES = (16, 32, 64, 128, 256)


def _pair_scores(cell, values, seed):
    scores = []
    for label in range(8):
        index = np.flatnonzero(cell == label)
        index = index[np.random.default_rng(seed + label).permutation(len(index))]
        index = index[:2 * (len(index) // 2)]
        scores.extend(.5 * np.abs(values[index[::2]] - values[index[1::2]]))
    return np.asarray(scores), 2 * len(scores) / len(values)


def main():
    rows = []
    n_clusters, repeats = 4096, 2
    for mechanism_index, (name, generator, base_floor) in enumerate(LAWS):
        for replicate in range(50):
            seed = 1_200_001 + 50 * mechanism_index + replicate
            rng = np.random.default_rng(seed)
            cluster_cell = rng.integers(0, 8, n_clusters)
            scale = .7 + .12 * cluster_cell
            location = 5. * cluster_cell
            cluster_outcome = location + scale * generator(rng, n_clusters)
            labels = np.repeat(cluster_cell, repeats)
            outcomes = np.repeat(cluster_outcome, repeats)
            clusters = np.repeat(np.arange(n_clusters), repeats)
            gold = float(base_floor * np.mean(scale))

            block = clustered_block_saa_portfolio(
                labels, outcomes, clusters, loss="mae",
                block_cluster_sizes=BLOCK_SIZES, score_caps=CAPS, alpha=.05,
                partition_seed=4401,
                representation_provenance="frozen-eight-cell-map-v1",
                partition_provenance="seeded-disjoint-patient-blocks-v1",
                independent_clusters_attested=True,
                outcome_blind_labels_attested=True,
                partition_frozen_before_outcomes=True)
            pair_scores, retained = _pair_scores(cluster_cell, cluster_outcome, 4401)
            pair = truncated_saa_portfolio(
                {"disjoint_within_cell_pairs": pair_scores}, loss="mae",
                score_caps=CAPS, alpha_family=.05,
                retained_fractions={"disjoint_within_cell_pairs": retained},
                route_provenance={
                    "partition": "seeded-disjoint-within-cell-pairs-v1",
                    "witness": "half absolute disagreement"},
                portfolio_frozen_before_evaluation=True)
            rows.append({
                "mechanism": name, "seed": seed, "n_clusters": n_clusters,
                "n_rows": len(outcomes), "true_floor": gold,
                "block_bound": block.bound, "pair_bound": pair.bound,
                "block_valid": bool(block.bound <= gold + 1e-12),
                "pair_valid": bool(pair.bound <= gold + 1e-12),
                "block_positive": bool(block.bound > 0),
                "block_strict_win": bool(block.bound > pair.bound + 1e-12),
                "block_selected_route": block.selected_route,
                "block_selected_cap": block.selected_score_cap,
                "pair_selected_cap": pair.selected_score_cap,
            })

    output = Path(__file__).resolve().parent / "support_free_cluster_block_results.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
