"""Post-confirmation development of cell-specific bounded differences."""
from __future__ import annotations

import json
import sys

import numpy as np

from peras import finite_cell_saa_lower_bound
import run_intervention_battery as base


def _range_map(transform, labels, original_cells, laws):
    result = {}
    for label in np.unique(labels):
        if transform in {"baseline", "nuisance_canonical", "token_shuffle_bow",
                         "bijective", "nuisance_naive"}:
            source = np.unique(original_cells[labels == label])
        elif transform in {"family_garble", "within_family_reassign"}:
            family = int(label) if transform == "family_garble" else int(label) // 4
            source = np.arange(4 * family, 4 * family + 4)
        elif transform == "global_misalign":
            source = np.arange(20)
        else:
            continue
        values = laws[source].reshape(-1)
        result[str(label)] = float(values.max() - values.min())
    return result


def run_one(name, seed):
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 20, 2048)
    outcomes = base._sample(name, cells, rng)
    certify = np.arange(2048) >= 1024
    laws, by_cell, by_family, global_risk = base._gold_tables(name)
    rows = []
    for j, transform in enumerate(base.TRANSFORMS[:8]):
        labels = base._labels(
            transform, cells, outcomes, np.random.default_rng(seed * 101 + j))
        selected_labels = labels[certify]
        selected_cells = cells[certify]
        ranges = _range_map(transform, selected_labels, selected_cells, laws)
        gold = base._gold(
            transform, selected_labels, selected_cells,
            by_cell, by_family, global_risk)
        result = finite_cell_saa_lower_bound(
            selected_labels, outcomes[certify], np.flatnonzero(certify), loss="mae",
            outcome_range_bound=None, alpha=.05,
            representation_provenance=f"synthetic-{transform}-v1",
            support_provenance=f"analytic-{name}-support-v1",
            cell_outcome_range_bounds=ranges,
            cell_support_provenance=f"analytic-{name}-{transform}-cell-support-v1",
            iid_rows_attested=True, outcome_blind_attested=True)
        rows.append({
            "mechanism": name, "seed": seed, "transform": transform,
            "gold": gold, "bound": result.bound,
            "ratio": result.bound / gold if gold > 0 else None,
            "violation": result.bound > gold + 1e-12,
            "width": result.concentration_width,
        })
    return rows


if __name__ == "__main__":
    output = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 61001
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism in base.MECHANISMS:
            for seed in range(start, start + count):
                for row in run_one(mechanism, seed):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(mechanism, flush=True)
