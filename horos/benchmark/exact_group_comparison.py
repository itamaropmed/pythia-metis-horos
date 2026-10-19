"""Fresh known-risk audit of exact group Lipschitz witnesses; no model training."""
from __future__ import annotations

import json

import numpy as np

from horos.represent.grouping import nearest_group_hierarchy
from horos.represent.pairing import global_nearest_pairs
from pythia.core.group_witness import grouped_decision_envelope
from pythia.core.replicates import paired_replicate_lower_bound

from .multiloss_comparison import LOSSES
from .multiloss_synthetic import MULTILOSS_DESIGNS, multiloss_design


def run_exact_group_comparison(
    *, seed: int = 20261008, repetitions: int = 5, n_entities: int = 256,
    loss_labels: tuple[str, ...] | None = None,
) -> dict:
    """Compare predeclared radial/exact groups and a paired Bentkus baseline.

    Each loss receives alpha=.01. Group size selection is covered within each
    group-only envelope; comparing two envelopes here is descriptive, not a
    free outcome-selected certificate.
    """
    if repetitions < 1 or n_entities < 32 or n_entities % 16:
        raise ValueError("repetitions >=1 and n_entities divisible by 16 required")
    if loss_labels is not None and (
        not loss_labels or len(set(loss_labels)) != len(loss_labels)
        or any(label not in {item[0] for item in LOSSES} for label in loss_labels)
    ):
        raise ValueError("loss_labels must be a nonempty unique subset of registered losses")
    selected_losses = tuple(
        item for item in LOSSES if loss_labels is None or item[0] in loss_labels
    )
    seeds = np.random.SeedSequence(seed).spawn(len(MULTILOSS_DESIGNS) * repetitions)
    rows = []
    cursor = 0
    for name in MULTILOSS_DESIGNS:
        for _ in range(repetitions):
            design = multiloss_design(
                np.random.default_rng(seeds[cursor]), name, n_entities,
            )
            cursor += 1
            x, y = design["representation"], design["outcomes"]
            entities = np.arange(n_entities)
            groupings = nearest_group_hierarchy(x, entities, max_group_size=16)
            pairs = global_nearest_pairs(x, entities)
            for label, loss, quantile, window in selected_losses:
                truth = float(np.mean(design[f"truth_{label}"]))
                common = dict(
                    loss=loss, quantile=quantile, window=window,
                    lipschitz_bound=design["lipschitz_cap"],
                    outcome_range_bound=300. if loss == "pinball" else None,
                    alpha=.01, methods=("conditional_bentkus",),
                    pinball_range_rule="finite_group",
                )
                radial = grouped_decision_envelope(
                    x, y, entities, groupings, witness_rule="radial", **common,
                )
                exact = grouped_decision_envelope(
                    x, y, entities, groupings,
                    witness_rule="exact_lipschitz", **common,
                )
                entity_mcd = grouped_decision_envelope(
                    x, y, entities, groupings,
                    witness_rule="exact_lipschitz",
                    **{**common, "methods": ("conditional_entity_mcd",)},
                )
                safe_selected = grouped_decision_envelope(
                    x, y, entities, groupings,
                    witness_rule="exact_lipschitz",
                    **{**common, "methods": (
                        "conditional_bentkus", "conditional_entity_mcd",
                    )},
                )
                paired = paired_replicate_lower_bound(
                    y[pairs.left_indices], y[pairs.right_indices], pairs.distances,
                    entities[pairs.left_indices], entities[pairs.right_indices],
                    pairs.pair_blocks, loss=loss, quantile=quantile, window=window,
                    lipschitz_bound=design["lipschitz_cap"], distance_bound=1.,
                    outcome_difference_bound=300. if loss == "pinball" else None,
                    alpha=.01, concentration="conditional_bentkus",
                )
                paired_classics = {
                    method: paired_replicate_lower_bound(
                        y[pairs.left_indices], y[pairs.right_indices],
                        pairs.distances, entities[pairs.left_indices],
                        entities[pairs.right_indices], pairs.pair_blocks,
                        loss=loss, quantile=quantile, window=window,
                        lipschitz_bound=design["lipschitz_cap"],
                        distance_bound=1.,
                        outcome_difference_bound=(
                            300. if loss == "pinball" else None
                        ),
                        alpha=.01, concentration=method,
                    ) for method in (
                        "conditional_hoeffding",
                        "conditional_empirical_bernstein",
                    )
                }
                if any(result["status"] != "complete" for result in
                       (radial, exact, entity_mcd, safe_selected, paired,
                        *paired_classics.values())):
                    raise RuntimeError("known-risk design unexpectedly refused")
                routed = (
                    paired if loss == "pinball" else
                    entity_mcd if label == "w15_miss" else exact
                )
                safe_route = paired if loss == "pinball" else safe_selected
                geometry_ratio = float(
                    design["lipschitz_cap"] * np.median(pairs.distances)
                    / (300. if loss == "pinball" else max(window, 1.))
                )
                geometry_route = (
                    exact if geometry_ratio <= .01 else paired
                ) if loss == "pinball" else (
                    safe_selected if geometry_ratio <= .5 else paired
                )
                classic_best = max(
                    paired["bound"],
                    *(result["bound"] for result in paired_classics.values()),
                )
                rows.append({
                    "design": name, "loss": label, "truth": truth,
                    "radial": radial["bound"], "exact": exact["bound"],
                    "entity_mcd": entity_mcd["bound"],
                    "safe_selected": safe_selected["bound"],
                    "paired": paired["bound"],
                    "paired_hoeffding": paired_classics[
                        "conditional_hoeffding"
                    ]["bound"],
                    "paired_empirical_bernstein": paired_classics[
                        "conditional_empirical_bernstein"
                    ]["bound"],
                    "classic_best_unadjusted": classic_best,
                    "routed": routed["bound"],
                    "safe_route": safe_route["bound"],
                    "geometry_route": geometry_route["bound"],
                    "geometry_ratio": geometry_ratio,
                    "geometry_family": (
                        "exact_group" if geometry_ratio <= .01 else "paired"
                    ) if loss == "pinball" else (
                        "safe_group" if geometry_ratio <= .5 else "paired"
                    ),
                    "radial_center": radial["estimate"],
                    "exact_center": exact["estimate"],
                })

    def summary(selected: list[dict]) -> dict:
        truth = np.array([row["truth"] for row in selected])
        positive = truth > 0
        answer = {"total": len(selected), "positive_floor": int(positive.sum())}
        for method in ("radial", "exact", "entity_mcd", "safe_selected",
                       "paired", "paired_hoeffding",
                       "paired_empirical_bernstein", "classic_best_unadjusted",
                       "routed", "safe_route", "geometry_route"):
            values = np.array([row[method] for row in selected])
            answer[method] = {
                "valid": int(np.sum(values <= truth + 1e-12)),
                "median_usable_tightness": float(np.median(
                    values[positive] / truth[positive],
                )) if np.any(positive) else None,
            }
        exact = np.array([row["exact"] for row in selected])
        for baseline in ("radial", "paired", "entity_mcd"):
            other = np.array([row[baseline] for row in selected])
            answer[f"exact_vs_{baseline}"] = {
                "strict_wins": int(np.sum(exact > other + 1e-10)),
                "ties": int(np.sum(np.abs(exact - other) <= 1e-10)),
                "strict_losses": int(np.sum(exact + 1e-10 < other)),
            }
        routed = np.array([row["routed"] for row in selected])
        paired = np.array([row["paired"] for row in selected])
        answer["routed_vs_paired"] = {
            "strict_wins": int(np.sum(routed > paired + 1e-10)),
            "ties": int(np.sum(np.abs(routed - paired) <= 1e-10)),
            "strict_losses": int(np.sum(routed + 1e-10 < paired)),
        }
        safe = np.array([row["safe_route"] for row in selected])
        answer["safe_route_vs_paired"] = {
            "strict_wins": int(np.sum(safe > paired + 1e-10)),
            "ties": int(np.sum(np.abs(safe - paired) <= 1e-10)),
            "strict_losses": int(np.sum(safe + 1e-10 < paired)),
        }
        geometry = np.array([row["geometry_route"] for row in selected])
        answer["geometry_route_vs_paired"] = {
            "strict_wins": int(np.sum(geometry > paired + 1e-10)),
            "ties": int(np.sum(np.abs(geometry - paired) <= 1e-10)),
            "strict_losses": int(np.sum(geometry + 1e-10 < paired)),
        }
        classic_best = np.array([row["classic_best_unadjusted"] for row in selected])
        answer["geometry_route_vs_classic_best_unadjusted"] = {
            "strict_wins": int(np.sum(geometry > classic_best + 1e-10)),
            "ties": int(np.sum(np.abs(geometry - classic_best) <= 1e-10)),
            "strict_losses": int(np.sum(geometry + 1e-10 < classic_best)),
        }
        answer["geometry_family_counts"] = {
            family: sum(row["geometry_family"] == family for row in selected)
            for family in ("exact_group", "paired", "safe_group")
        }
        return answer

    return {
        "status": "exploratory_known_risk_comparison",
        "seed": seed, "repetitions_per_design": repetitions,
        "n_entities": n_entities, "n_designs": len(MULTILOSS_DESIGNS),
        "loss_registry": [item[0] for item in selected_losses],
        "pythia_trained_predictive_model": False,
        "selection_between_witnesses_adjusted": False,
        "loss_only_route": {
            "q10_pinball": "paired_bentkus", "q90_pinball": "paired_bentkus",
            "w15_miss": "exact_group_entity_mcd",
            "w30_miss": "exact_group_bentkus",
            "w60_miss": "exact_group_bentkus",
        },
        "safe_route": "paired Bentkus for pinball; Bonferroni Bentkus/entity-McDiarmid exact-group selection for windows",
        "geometry_route": "pinball exact group iff median declared pair action variation / outcome span <=0.01, else paired; safe-group windows iff median variation / window <=0.5, else paired",
        "summary": summary(rows),
        "by_loss": {
            label: summary([row for row in rows if row["loss"] == label])
            for label, _, _, _ in selected_losses
        },
        "by_design": {
            name: summary([row for row in rows if row["design"] == name])
            for name in MULTILOSS_DESIGNS
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_exact_group_comparison(), indent=2, sort_keys=True))
