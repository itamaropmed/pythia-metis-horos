"""Frozen Monte Carlo audit for Pythia's clustered local-risk certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pythia.core.local_model_risk import build_local_model_risk_registry


def run(repetitions: int = 100, seed: int = 20261018) -> dict:
    rng = np.random.default_rng(seed)
    cap = 100.0
    patients = 400
    entity = np.repeat(np.arange(patients), 2)
    entity_cell = np.where(np.arange(patients) < patients // 2, "low", "high")
    cell = np.repeat(entity_cell, 2)
    global_label = np.repeat("all", len(entity))
    probabilities = {
        "basic": {"low": .20, "high": .65},
        "plus11": {"low": .15, "high": .50},
    }
    covered_families = 0
    false_relations = 0
    relation_counts: dict[str, int] = {}
    widths = []
    for _ in range(repetitions):
        losses = {}
        for model, probability in probabilities.items():
            # One latent draw per patient makes the two repeated rows perfectly
            # dependent.  Treating 800 rows as independent would be invalid.
            latent = rng.random(patients)
            patient_loss = cap * np.array([
                latent[i] < probability[entity_cell[i]] for i in range(patients)
            ])
            losses[model] = np.repeat(patient_loss, 2)
        registry = build_local_model_risk_registry(
            losses, {"cell": cell, "global": global_label}, entity,
            loss_cap=cap, alpha=.05, min_entities=100,
            outcome_blind_hierarchy=True, independent_entities=True,
            loss_cap_attested=True,
        )
        family_ok = True
        for model, probability in probabilities.items():
            truth = {
                "cell|low": cap * probability["low"],
                "cell|high": cap * probability["high"],
                "global|all": cap * (probability["low"] + probability["high"]) / 2,
            }
            for suffix, target in truth.items():
                node = registry.nodes[f"{model}|{suffix}"]
                family_ok &= node.lower <= target <= node.upper
                widths.append(node.upper - node.lower)
            for key in ("low", "high"):
                routed = registry.route(model, {"cell": key, "global": "all"})
                relation_counts[routed.local_relation] = (
                    relation_counts.get(routed.local_relation, 0) + 1
                )
                true_local = truth[f"cell|{key}"]
                true_global = truth["global|all"]
                if (routed.local_relation == "certified_higher_than_global"
                        and not true_local > true_global):
                    false_relations += 1
                if (routed.local_relation == "certified_lower_than_global"
                        and not true_local < true_global):
                    false_relations += 1
        covered_families += int(family_ok)
    return {
        "design": "two rows per patient with perfect within-patient dependence",
        "repetitions": repetitions,
        "alpha_family": .05,
        "simultaneous_family_coverage": covered_families / repetitions,
        "covered_families": covered_families,
        "false_certified_relations": false_relations,
        "relation_counts": relation_counts,
        "median_interval_width": float(np.median(widths)),
        "seed": seed,
        "claim": "development Monte Carlo audit; not confirmatory evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.repetitions, args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
