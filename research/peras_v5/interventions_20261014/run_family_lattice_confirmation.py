"""Frozen runner for FAMILY_LATTICE_CONFIRMATION_REGISTRATION.md."""
from __future__ import annotations

import json
import sys

import numpy as np

from peras import finite_cell_information_lattice
import run_intervention_battery as base


def _risk(values):
    return float(np.mean(np.abs(values - np.median(values))))


def _laws(name):
    residual = base._laws(name)
    scale = np.maximum(np.ptp(residual, axis=1), .1)
    shift = .18 * scale
    return np.stack((residual - shift[:, None], residual + shift[:, None]), axis=1)


def _tables(name):
    laws = _laws(name)
    neighborhood = np.empty((20, 2))
    procedure = np.empty(20)
    family = np.empty(5)
    for cell in range(20):
        for concept in range(2):
            neighborhood[cell, concept] = _risk(laws[cell, concept])
        procedure[cell] = _risk(laws[cell].reshape(-1))
    for group in range(5):
        family[group] = _risk(laws[4 * group:4 * group + 4].reshape(-1))
    return laws, neighborhood, procedure, family, _risk(laws.reshape(-1))


def _ranges(labels, sources, laws):
    result = {}
    for label in np.unique(labels):
        selected = sources[label]
        values = np.concatenate([laws[cell, concept].reshape(-1)
                                 for cell, concept in selected])
        result[str(label)] = float(np.ptp(values))
    return result


def run_one(mechanism, seed):
    rng = np.random.default_rng(seed)
    n = 2048
    cell = rng.integers(0, 20, n)
    concept = rng.integers(0, 2, n)
    nuisance = rng.integers(0, 16, n)
    laws, gold_neighborhood, gold_procedure, gold_family, gold_global = _tables(mechanism)
    index = np.minimum((rng.random(n) * laws.shape[-1]).astype(int), laws.shape[-1] - 1)
    y = laws[cell, concept, index]

    global_label = np.repeat("all", n)
    family_label = (cell // 4).astype(str)
    procedure_label = cell.astype(str)
    neighborhood_label = np.array([f"{c}|{z}" for c, z in zip(cell, concept)])
    nuisance_label = np.array([f"{c}|{z}|{r}" for c, z, r
                               in zip(cell, concept, nuisance)])
    representations = {
        "global": global_label,
        "family": family_label,
        "procedure": procedure_label,
        "neighborhood": neighborhood_label,
        "nuisance": nuisance_label,
    }
    global_sources = {"all": [(c, z) for c in range(20) for z in range(2)]}
    family_sources = {str(g): [(c, z) for c in range(4 * g, 4 * g + 4)
                               for z in range(2)] for g in range(5)}
    procedure_sources = {str(c): [(c, z) for z in range(2)] for c in range(20)}
    neighborhood_sources = {f"{c}|{z}": [(c, z)] for c in range(20) for z in range(2)}
    nuisance_sources = {label: [(int(label.split("|")[0]), int(label.split("|")[1]))]
                        for label in np.unique(nuisance_label)}
    cap_sources = {
        "global": global_sources,
        "family": family_sources,
        "procedure": procedure_sources,
        "neighborhood": neighborhood_sources,
        "nuisance": nuisance_sources,
    }
    caps = {name: _ranges(labels, cap_sources[name], laws)
            for name, labels in representations.items()}
    graph = (("nuisance", "neighborhood"), ("neighborhood", "procedure"),
             ("procedure", "family"), ("family", "global"))
    result = finite_cell_information_lattice(
        representations, y, np.arange(n), richer_to_poorer=graph,
        equal_risk_pairs=(("nuisance", "neighborhood"),), loss="mae",
        outcome_range_bound=None, alpha=.05,
        representation_provenance={name: f"synthetic-{name}-v1"
                                   for name in representations},
        support_provenance=f"analytic-{mechanism}-support-v1",
        nesting_provenance="registered-family-neighborhood-maps-v1",
        equivalence_provenance="seeded-independent-nuisance-v1",
        iid_rows_attested=True, outcome_blind_attested=True,
        nesting_attested=True, equivalence_attested=True,
        cell_outcome_range_bounds=caps,
        cell_support_provenance={name: f"analytic-{mechanism}-{name}-support-v1"
                                 for name in representations})
    gold = {
        "global": float(gold_global),
        "family": float(np.mean(gold_family[cell // 4])),
        "procedure": float(np.mean(gold_procedure[cell])),
        "neighborhood": float(np.mean(gold_neighborhood[cell, concept])),
        "nuisance": float(np.mean(gold_neighborhood[cell, concept])),
    }
    raw = {name: result.node_results[name].bound for name in representations}
    raw_estimate = {name: result.node_results[name].empirical_saa_score
                    for name in representations}
    closed = {name: result.order_result.intervals[name].lower for name in representations}
    estimates = {name: result.order_result.intervals[name].estimate for name in representations}
    return {
        "mechanism": mechanism, "seed": seed, "status": result.status,
        "gold": gold, "raw_bound": raw, "closed_bound": closed,
        "raw_estimate": raw_estimate,
        "ordered_estimate": estimates,
        "equality_estimator_source": result.diagnostics["equality_estimator_source"],
        "violation": any(raw[name] > gold[name] + 1e-12 for name in representations),
        "closed_violation": any(closed[name] > gold[name] + 1e-12
                                for name in representations),
        "nuisance_repaired": abs(closed["nuisance"] - closed["neighborhood"]) <= 1e-12,
        "nuisance_raw_weaker": raw["nuisance"] < raw["neighborhood"] - 1e-12,
        "order_ok": (closed["nuisance"] <= closed["neighborhood"] + 1e-12
                     and closed["neighborhood"] <= closed["procedure"] + 1e-12
                     and closed["procedure"] <= closed["family"] + 1e-12
                     and closed["family"] <= closed["global"] + 1e-12),
        "spent_alpha": result.spent_alpha,
        "model_training": result.diagnostics["model_training_performed"],
    }


if __name__ == "__main__":
    output = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 63001
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism in base.MECHANISMS:
            for seed in range(start, start + count):
                handle.write(json.dumps(run_one(mechanism, seed), sort_keys=True) + "\n")
            print(mechanism, flush=True)
