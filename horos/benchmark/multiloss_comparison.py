"""Known-risk comparison across decision losses, numeric and note-like inputs."""
from __future__ import annotations

import numpy as np

from horos.represent.grouping import nearest_group_hierarchy
from horos.represent.pairing import global_nearest_pairs
from pythia.core.decision_order import monotone_window_floor_closure
from pythia.core.mae_family_envelope import decision_pair_group_envelope
from pythia.core.replicates import paired_replicate_lower_bound

from .decision_plugins import crossfit_local_decision_plugins
from .dependence_comparison import _wilson_lower
from .multiloss_synthetic import MULTILOSS_DESIGNS, multiloss_design


LOSSES = (
    ("q10_pinball", "pinball", .1, None),
    ("q90_pinball", "pinball", .9, None),
    ("w15_miss", "window_miss", None, 15.),
    ("w30_miss", "window_miss", None, 30.),
    ("w60_miss", "window_miss", None, 60.),
)
METHODS = (
    "conditional_hoeffding", "conditional_empirical_bernstein",
    "conditional_bentkus",
)


def run_multiloss_comparison(
    *, seed: int = 20261004, repetitions: int = 10, n_entities: int = 512,
) -> dict:
    """Compare three classic concentration bounds and three external plugins.

    Each bound receives alpha=.05/5, ensuring joint 95% over the five
    predeclared losses for one design/draw under independent-entity conditions.
    No claim of joint confidence across repeated designs/draws is made.
    """
    if repetitions < 1 or n_entities < 32 or n_entities % 16:
        raise ValueError("positive repetitions and n_entities divisible by 16 >=32 required")
    seeds = np.random.SeedSequence(seed).spawn(len(MULTILOSS_DESIGNS) * repetitions)
    rows = []
    cursor = 0
    for design_name in MULTILOSS_DESIGNS:
        for _ in range(repetitions):
            draw_id = cursor
            design = multiloss_design(
                np.random.default_rng(seeds[cursor]), design_name, n_entities,
            )
            cursor += 1
            x, y = design["representation"], design["outcomes"]
            entities = np.arange(n_entities)
            pairs = global_nearest_pairs(x, entities)
            groupings = nearest_group_hierarchy(x, entities, max_group_size=16)
            plugins = crossfit_local_decision_plugins(x, y, seed=draw_id + 101)
            for label, loss, quantile, window in LOSSES:
                truth = float(np.mean(design[f"truth_{label}"]))
                paired_args = (
                    y[pairs.left_indices], y[pairs.right_indices], pairs.distances,
                    entities[pairs.left_indices], entities[pairs.right_indices],
                    pairs.pair_blocks,
                )
                kwargs = dict(
                    loss=loss, quantile=quantile, window=window,
                    lipschitz_bound=design["lipschitz_cap"],
                    distance_bound=1.,
                    outcome_difference_bound=300. if loss == "pinball" else None,
                    alpha=.05 / len(LOSSES),
                )
                baselines = {
                    method: paired_replicate_lower_bound(
                        *paired_args, concentration=method, **kwargs,
                    ) for method in METHODS
                }
                if any(value["status"] != "complete" for value in baselines.values()):
                    raise RuntimeError("known bounded DGP was refused by a paired comparator")
                combined = decision_pair_group_envelope(
                    x, y, entities, pairs.left_indices, pairs.right_indices,
                    groupings, loss=loss, quantile=quantile, window=window,
                    lipschitz_bound=design["lipschitz_cap"],
                    outcome_range_bound=300. if loss == "pinball" else None,
                    alpha=.05 / len(LOSSES), methods=METHODS,
                )
                if combined["status"] != "complete":
                    raise RuntimeError("known bounded DGP was refused by combined envelope")
                # Frozen development choice: pair geometry for quantiles,
                # multi-outcome group geometry for window misses. Routing is
                # loss-only and never reads outcomes on the evaluation draw.
                routed_family = "paired" if loss == "pinball" else "grouped"
                routed = decision_pair_group_envelope(
                    x, y, entities, pairs.left_indices, pairs.right_indices,
                    groupings, loss=loss, quantile=quantile, window=window,
                    lipschitz_bound=design["lipschitz_cap"],
                    outcome_range_bound=300. if loss == "pinball" else None,
                    alpha=.05 / len(LOSSES), family_policy=routed_family,
                    methods=("conditional_bentkus",),
                )
                if routed["status"] != "complete":
                    raise RuntimeError("known bounded DGP refused routed family")
                routed_bound = routed["bound"]
                rows.append({
                    "design": design_name, "draw_id": draw_id,
                    "input_type": "preoutcome_note_text" if design["notes_preoutcome"] else "numeric",
                    "loss": label, "truth": truth,
                    "combined": combined["bound"],
                    "combined_center": combined["estimate"],
                    "selected_family": combined["selected_family"],
                    "routed_bentkus": routed_bound,
                    "routed_family": routed_family,
                    **{method: result["bound"] for method, result in baselines.items()},
                    **{method: plugins["methods"][method][label]
                       for method in plugins["methods"]},
                })
            window_rows = {
                width: next(row for row in rows[-len(LOSSES):]
                            if row["loss"] == f"w{width}_miss")
                for width in (15, 30, 60)
            }
            for method in ("combined", "routed_bentkus", *METHODS):
                closed = monotone_window_floor_closure({
                    width: row[method] for width, row in window_rows.items()
                })
                for width, bound in closed.items():
                    window_rows[int(width)][method] = bound

    def summarize(selected: list[dict]) -> dict:
        if not selected:
            return {}
        truth = np.asarray([row["truth"] for row in selected])
        answer = {"n_loss_results": len(selected),
                  "n_independent_draws": len({row["draw_id"] for row in selected}),
                  "n_zero_true_floor_results": int(np.sum(truth == 0))}
        for method in ("combined", "routed_bentkus", *METHODS):
            values = np.asarray([row[method] for row in selected])
            valid = values <= truth + 1e-12
            positive_truth = truth > 0
            answer[method] = {
                "valid": int(np.sum(valid)), "total": len(selected),
                "median_usable_tightness": (
                    float(np.median(np.where(
                        valid[positive_truth],
                        values[positive_truth] / truth[positive_truth], 0,
                    ))) if np.any(positive_truth) else None
                ),
                "tightness_denominator": int(np.sum(positive_truth)),
            }
        joint = [
            all(row["combined"] <= row["truth"] + 1e-12
                for row in selected if row["draw_id"] == draw)
            for draw in sorted({row["draw_id"] for row in selected})
        ]
        answer["combined_joint_draw_valid"] = int(sum(joint))
        answer["combined_joint_draw_wilson_lower_95"] = _wilson_lower(
            int(sum(joint)), len(joint),
        )
        routed_joint = [
            all(row["routed_bentkus"] <= row["truth"] + 1e-12
                for row in selected if row["draw_id"] == draw)
            for draw in sorted({row["draw_id"] for row in selected})
        ]
        answer["routed_joint_draw_valid"] = int(sum(routed_joint))
        combined = np.asarray([row["combined"] for row in selected])
        routed = np.asarray([row["routed_bentkus"] for row in selected])
        answer["bound_comparisons"] = {
            method: {
                "combined_strict_wins": int(np.sum(combined > np.asarray([
                    row[method] for row in selected]) + 1e-12)),
                "baseline_strict_wins": int(np.sum(combined + 1e-12 < np.asarray([
                    row[method] for row in selected]))),
                "routed_strict_wins": int(np.sum(routed > np.asarray([
                    row[method] for row in selected]) + 1e-12)),
                "baseline_strict_wins_over_routed": int(np.sum(
                    routed + 1e-12 < np.asarray([row[method] for row in selected])
                )),
            } for method in METHODS
        }
        answer["median_absolute_point_error"] = {
            method: float(np.median(np.abs(np.asarray([
                row[method] for row in selected]) - truth)))
            for method in ("combined_center", "knn", "histogram", "kernel")
        }
        return answer

    return {
        "status": "exploratory_known_risk_comparison",
        "seed": seed, "repetitions_per_design": repetitions,
        "n_entities": n_entities,
        "design_registry": list(MULTILOSS_DESIGNS),
        "loss_registry": [label for label, _, _, _ in LOSSES],
        "bound_comparators": list(METHODS),
        "loss_only_routing": {
            label: "paired_bentkus" if loss == "pinball" else "grouped_bentkus"
            for label, loss, _, _ in LOSSES
        },
        "external_point_estimators": ["knn", "histogram", "kernel"],
        "pythia_trained_predictive_model": False,
        "simultaneous_window_order_closure": True,
        "external_estimators_cross_fitted": True,
        "confidence_per_draw_simultaneous_across_losses": .95,
        "summary": summarize(rows),
        "by_loss": {
            label: summarize([row for row in rows if row["loss"] == label])
            for label, _, _, _ in LOSSES
        },
        "by_input_type": {
            kind: summarize([row for row in rows if row["input_type"] == kind])
            for kind in ("numeric", "preoutcome_note_text")
        },
        "by_design": {
            name: summarize([row for row in rows if row["design"] == name])
            for name in MULTILOSS_DESIGNS
        },
    }
