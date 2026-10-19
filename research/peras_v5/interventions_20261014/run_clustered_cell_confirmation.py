#!/usr/bin/env python3
"""Frozen confirmation for CLUSTERED_CELL_CONFIRMATION_REGISTRATION.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from peras import clustered_finite_cell_saa_lower_bound, finite_cell_saa_lower_bound


LAWS = (
    ("rademacher", (-1., 1.), (.5, .5)),
    ("center_heavy", (-1., 0., 1.), (.2, .6, .2)),
    ("tail_heavy", (-1., 0., 1.), (.45, .1, .45)),
    ("right_skew", (-1., 0., .5, 1.), (.1, .55, .2, .15)),
    ("left_skew", (-1., -.5, 0., 1.), (.15, .2, .55, .1)),
    ("rare_tail", (-1., 0., 1.), (.05, .9, .05)),
    ("five_uniform", (-1., -.5, 0., .5, 1.), (.2, .2, .2, .2, .2)),
    ("bimodal_center", (-1., -.75, .75, 1.), (.25, .25, .25, .25)),
    ("asymmetric_mixture", (-1., -.2, .1, .9), (.2, .3, .35, .15)),
    ("nine_uniform", tuple(np.linspace(-1., 1., 9)), tuple(np.ones(9) / 9)),
)


def exact_mae_floor(values, probabilities):
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    median = values[np.searchsorted(np.cumsum(probabilities), .5)]
    return float(np.sum(probabilities * np.abs(values - median)))


def _common(labels, outcomes, *, outcome_range_bound, cell_ranges=None):
    return dict(
        cell_labels=labels, outcomes=outcomes, loss="mae",
        outcome_range_bound=outcome_range_bound,
        cell_outcome_range_bounds=cell_ranges,
        cell_support_provenance=(
            "analytic conditional support from frozen discrete law"
            if cell_ranges is not None else None),
        alpha=.05, representation_provenance="frozen-eight-cell-map-v1",
        support_provenance="analytic bounded population support",
        outcome_blind_attested=True)


def main():
    here = Path(__file__).resolve().parent
    output = here / "clustered_cell_confirmation_results.jsonl"
    rows = []
    for mechanism_index, (name, raw_values, raw_probabilities) in enumerate(LAWS):
        values = np.asarray(raw_values, dtype=float)
        probabilities = np.asarray(raw_probabilities, dtype=float)
        base_floor = exact_mae_floor(values, probabilities)
        for replicate in range(100):
            seed = 760001 + 100 * mechanism_index + replicate
            rng = np.random.default_rng(seed)
            n_clusters, repeats = 80, 12
            cluster_cell = rng.integers(0, 8, n_clusters)
            scales = .7 + .12 * cluster_cell
            locations = 5. * cluster_cell
            residual = rng.choice(values, n_clusters, p=probabilities)
            labels = np.repeat(cluster_cell, repeats)
            clusters = np.repeat(np.arange(n_clusters), repeats)
            outcomes = np.repeat(locations + scales * residual, repeats)
            cell_scales = .7 + .12 * np.arange(8)
            cell_ranges = {
                str(cell): float(cell_scales[cell] * (values.max() - values.min()))
                for cell in range(8)}
            true_floor = float(np.mean((.7 + .12 * labels) * base_floor))
            support_low = float(np.min(5. * np.arange(8) + cell_scales * values.min()))
            support_high = float(np.max(5. * np.arange(8) + cell_scales * values.max()))
            global_range = support_high - support_low

            local = clustered_finite_cell_saa_lower_bound(
                clusters=clusters, independent_clusters_attested=True,
                **_common(labels, outcomes, outcome_range_bound=None,
                          cell_ranges=cell_ranges))
            global_result = clustered_finite_cell_saa_lower_bound(
                clusters=clusters, independent_clusters_attested=True,
                **_common(labels, outcomes, outcome_range_bound=global_range))
            invalid_row = finite_cell_saa_lower_bound(
                entities=np.arange(len(outcomes)), iid_rows_attested=True,
                **_common(labels, outcomes, outcome_range_bound=None,
                          cell_ranges=cell_ranges))
            rows.append({
                "study": "known_law", "mechanism": name, "seed": seed,
                "n_clusters": n_clusters, "n_rows": len(outcomes),
                "true_floor": true_floor, "local_cluster_bound": local.bound,
                "global_cluster_bound": global_result.bound,
                "invalid_row_iid_bound": invalid_row.bound,
                "local_cluster_valid": bool(local.bound <= true_floor + 1e-12),
                "global_cluster_valid": bool(global_result.bound <= true_floor + 1e-12),
                "invalid_row_iid_valid": bool(invalid_row.bound <= true_floor + 1e-12),
                "local_cluster_positive": bool(local.bound > 0),
                "local_strictly_beats_global": bool(local.bound > global_result.bound + 1e-12),
            })

    stress_values = np.asarray((-1., 0., 1.))
    stress_probabilities = np.asarray((.2, .6, .2))
    for replicate in range(1000):
        seed = 880001 + replicate
        rng = np.random.default_rng(seed)
        n_clusters, repeats = 60, 100
        residual = rng.choice(stress_values, n_clusters, p=stress_probabilities)
        labels = np.zeros(n_clusters * repeats, dtype=int)
        clusters = np.repeat(np.arange(n_clusters), repeats)
        outcomes = np.repeat(residual, repeats)
        common = _common(labels, outcomes, outcome_range_bound=2.)
        clustered = clustered_finite_cell_saa_lower_bound(
            clusters=clusters, independent_clusters_attested=True, **common)
        invalid_row = finite_cell_saa_lower_bound(
            entities=np.arange(len(outcomes)), iid_rows_attested=True, **common)
        rows.append({
            "study": "dependence_stress", "mechanism": "center_heavy",
            "seed": seed, "n_clusters": n_clusters, "n_rows": len(outcomes),
            "true_floor": .4, "local_cluster_bound": clustered.bound,
            "invalid_row_iid_bound": invalid_row.bound,
            "local_cluster_valid": bool(clustered.bound <= .4 + 1e-12),
            "invalid_row_iid_valid": bool(invalid_row.bound <= .4 + 1e-12),
        })

    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
