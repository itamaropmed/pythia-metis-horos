"""Known-truth stress test for task-conditioned pair-risk witnesses.

The two losses deliberately have different Bayes actions.  Truth is evaluated
at each audited covariate, not at an unrelated population average.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from horos.represent.pairing import global_nearest_pairs
from horos.represent.grouping import nearest_group_hierarchy
from pythia.core.group_witness import grouped_decision_envelope
from pythia.core.replicates import (
    paired_replicate_concentration_envelope, paired_replicate_lower_bound,
)

from .dependence_comparison import _wilson_lower
from .replicate_adversarial import DESIGN_NAMES, _design


def _knn_oof_risks(x: np.ndarray, y: np.ndarray, *, k: int = 51, window: float = .1) -> dict:
    """External leave-one-out kNN plug-in baseline, never used by Pythia."""
    if k >= len(y):
        k = len(y) - 1
    _, indices = cKDTree(x).query(x, k=k + 1)
    neighbors = y[indices[:, 1:]]
    q = np.quantile(neighbors, .9, axis=1)
    residual = y - q
    pinball = float(np.mean(np.maximum(.9 * residual, -.1 * residual)))
    # The optimal fixed-width window center on a finite neighbor sample can
    # be found by sorting and a two-pointer scan. This is a fitted benchmark,
    # explicitly outside the model-free Pythia certification path.
    centers = np.empty(len(y), float)
    for i, row in enumerate(neighbors):
        ordered = np.sort(row)
        left = best_left = best_right = 0
        for right in range(len(ordered)):
            while ordered[right] - ordered[left] > 2.0 * window:
                left += 1
            if right - left > best_right - best_left:
                best_left, best_right = left, right
        centers[i] = .5 * (ordered[best_left] + ordered[best_right])
    miss = float(np.mean(np.abs(y - centers) > window))
    return {"pinball": pinball, "window_miss": miss}


def _group_jackknife_risk(y: np.ndarray, groups: np.ndarray, *, loss: str) -> float:
    """Train-free exploratory bias correction of local empirical Bayes scores."""
    def score(values: np.ndarray) -> float:
        if loss == "pinball":
            q = float(np.quantile(values, .9))
            residual = values - q
            return float(np.mean(np.maximum(.9 * residual, -.1 * residual)))
        ordered = np.sort(values)
        left = best = 0
        for right in range(len(ordered)):
            while ordered[right] - ordered[left] > .2:
                left += 1
            best = max(best, right - left + 1)
        return float(1 - best / len(values))

    estimates = []
    for rows in groups:
        values = y[rows]
        mid = len(values) // 2
        estimates.append(2 * score(values) - .5 * (
            score(values[:mid]) + score(values[mid:])
        ))
    result = float(np.mean(estimates))
    return float(np.clip(result, 0, 1)) if loss == "window_miss" else float(max(result, 0))


def _truth(name: str, x: np.ndarray, loss: str) -> np.ndarray:
    n = len(x)
    if loss == "pinball":
        if name == "heteroskedastic":
            return .09 * (.2 + .6 * x[:, 0])
        values = {
            "high_dimensional": .054,
            "asymmetric_mixture": .06,
            "bounded_heavy_tail": .04,
            "bimodal_noise": .05,
        }
        return np.full(n, values.get(name, .045))
    if loss == "window_miss":
        if name == "heteroskedastic":
            return 1.0 - .1 / (.2 + .6 * x[:, 0])
        values = {
            "high_dimensional": 1.0 - .1 / .6,
            "asymmetric_mixture": .25,
            "bounded_heavy_tail": .04,
            "bimodal_noise": .5,
        }
        return np.full(n, values.get(name, .8))
    raise ValueError(loss)


def run_decision_loss_comparison(
    *, seed: int = 20260919, repetitions: int = 20, n_entities: int = 2000,
) -> dict:
    """Compare Hoeffding and empirical Bernstein on ten frozen DGP families."""
    if repetitions < 1 or n_entities < 4 or n_entities % 2:
        raise ValueError("positive repetitions and even n_entities >= 4 required")
    seeds = np.random.SeedSequence(seed).spawn(len(DESIGN_NAMES) * repetitions)
    rows = []
    cursor = 0
    for name in DESIGN_NAMES:
        for _ in range(repetitions):
            draw_id = cursor
            rng = np.random.default_rng(seeds[cursor]); cursor += 1
            x, y, base_l, _, difference_cap, _ = _design(rng, name, n_entities)
            entities = np.arange(n_entities)
            pairs = global_nearest_pairs(x, entities)
            hierarchy = nearest_group_hierarchy(x, entities, max_group_size=16)
            knn_risks = _knn_oof_risks(x, y)
            args = (
                y[pairs.left_indices], y[pairs.right_indices], pairs.distances,
                entities[pairs.left_indices], entities[pairs.right_indices],
                pairs.pair_blocks,
            )
            for loss in ("pinball", "window_miss"):
                # The heteroskedastic .9 quantile has derivative 1 + .6*.8.
                selected_l = 1.48 if name == "heteroskedastic" and loss == "pinball" else base_l
                kwargs = dict(
                    loss=loss, lipschitz_bound=selected_l,
                    distance_bound=1.0,
                    outcome_difference_bound=(difference_cap if loss == "pinball" else None),
                    quantile=(.9 if loss == "pinball" else None),
                    window=(.1 if loss == "window_miss" else None),
                )
                hf = paired_replicate_lower_bound(
                    *args, **kwargs, concentration="conditional_hoeffding",
                )
                eb = paired_replicate_lower_bound(
                    *args, **kwargs,
                    concentration="conditional_empirical_bernstein",
                )
                envelope = paired_replicate_concentration_envelope(
                    *args, **kwargs,
                )
                paired_half = paired_replicate_concentration_envelope(
                    *args, **kwargs, alpha=.025,
                )
                grouped = grouped_decision_envelope(
                    x, y, entities, hierarchy, loss=loss,
                    lipschitz_bound=selected_l,
                    outcome_range_bound=(difference_cap if loss == "pinball" else None),
                    quantile=(.9 if loss == "pinball" else None),
                    window=(.1 if loss == "window_miss" else None),
                    alpha=.05,
                )
                grouped_half = grouped_decision_envelope(
                    x, y, entities, hierarchy, loss=loss,
                    lipschitz_bound=selected_l,
                    outcome_range_bound=(difference_cap if loss == "pinball" else None),
                    quantile=(.9 if loss == "pinball" else None),
                    window=(.1 if loss == "window_miss" else None),
                    alpha=.025,
                )
                if grouped["status"] != "complete":
                    raise RuntimeError("known-risk group design violated its declared cap")
                conditional_truth = float(np.mean(_truth(name, x, loss)))
                largest_groups = hierarchy[max(hierarchy)]
                group_jackknife = _group_jackknife_risk(
                    y, largest_groups, loss=loss,
                )
                rows.append({
                    "design": name, "draw_id": draw_id,
                    "loss": loss, "truth": conditional_truth,
                    "hoeffding": hf["bound"], "empirical_bernstein": eb["bound"],
                    "selection_safe_envelope": envelope["bound"],
                    "selection_safe_grouped": grouped["bound"],
                    "selection_safe_combined": max(
                        paired_half["bound"], grouped_half["bound"]
                    ),
                    "selected_combined_family": (
                        "grouped" if grouped_half["bound"] > paired_half["bound"]
                        else "paired"
                    ),
                    "group_center": grouped["estimate"],
                    "selected_group_size": grouped["group_size"],
                    "selected_method": envelope["selected_method"],
                    "pair_center": eb["estimate"],
                    "knn_oof_plugin": knn_risks[loss],
                    "group_jackknife_plugin": group_jackknife,
                })

    def summarize(selected: list[dict]) -> dict:
        truth = np.asarray([row["truth"] for row in selected])
        output = {}
        for method in (
            "hoeffding", "empirical_bernstein", "selection_safe_envelope",
            "selection_safe_grouped",
            "selection_safe_combined",
        ):
            bounds = np.asarray([row[method] for row in selected])
            valid = bounds <= truth + 1e-12
            clustered_valid = {
                draw: all(bool(is_valid) for row, is_valid in zip(selected, valid)
                          if row["draw_id"] == draw)
                for draw in {row["draw_id"] for row in selected}
            }
            n_draws = len(clustered_valid)
            valid_draws = sum(clustered_valid.values())
            output[method] = {
                "valid": int(valid.sum()), "total": int(valid.size),
                "independent_draws_with_all_reported_losses_valid": int(valid_draws),
                "n_independent_draws": int(n_draws),
                "draw_cluster_wilson_lower_95": _wilson_lower(valid_draws, n_draws),
                "median_usable_tightness": float(np.median(np.where(valid, bounds / truth, 0))),
            }
        hf = np.asarray([row["hoeffding"] for row in selected])
        eb = np.asarray([row["empirical_bernstein"] for row in selected])
        center = np.asarray([row["pair_center"] for row in selected])
        group_center = np.asarray([row["group_center"] for row in selected])
        knn = np.asarray([row["knn_oof_plugin"] for row in selected])
        jackknife = np.asarray([row["group_jackknife_plugin"] for row in selected])
        paired_envelope = np.asarray([row["selection_safe_envelope"] for row in selected])
        grouped_envelope = np.asarray([row["selection_safe_grouped"] for row in selected])
        output["comparison"] = {
            "empirical_bernstein_strict_wins": int(np.sum(eb > hf + 1e-12)),
            "hoeffding_strict_wins": int(np.sum(hf > eb + 1e-12)),
            "median_pair_center_absolute_error": float(np.median(np.abs(center - truth))),
            "median_group_center_absolute_error": float(np.median(np.abs(group_center - truth))),
            "median_knn_plugin_absolute_error": float(np.median(np.abs(knn - truth))),
            "median_group_jackknife_absolute_error": float(np.median(np.abs(jackknife - truth))),
            "group_jackknife_closer_than_knn": int(np.sum(
                np.abs(jackknife - truth) + 1e-12 < np.abs(knn - truth)
            )),
            "group_center_closer_than_pair_center": int(np.sum(
                np.abs(group_center - truth) + 1e-12 < np.abs(center - truth)
            )),
            "group_center_closer_than_knn": int(np.sum(
                np.abs(group_center - truth) + 1e-12 < np.abs(knn - truth)
            )),
            "pair_center_closer_than_knn": int(np.sum(
                np.abs(center - truth) + 1e-12 < np.abs(knn - truth)
            )),
            "knn_closer_than_pair_center": int(np.sum(
                np.abs(knn - truth) + 1e-12 < np.abs(center - truth)
            )),
            "median_bound_gap_to_truth": float(np.median(truth - np.maximum(hf, eb))),
            "selection_safe_envelope_ties_or_wins_at_nominal_alpha": int(np.sum(
                np.asarray([row["selection_safe_envelope"] for row in selected])
                >= np.maximum(hf, eb) - 1e-12
            )),
            "grouped_strict_wins_over_paired_envelope": int(np.sum(
                grouped_envelope > paired_envelope + 1e-12
            )),
            "paired_strict_wins_over_grouped_envelope": int(np.sum(
                paired_envelope > grouped_envelope + 1e-12
            )),
            "median_grouped_minus_paired": float(np.median(
                grouped_envelope - paired_envelope
            )),
        }
        return output

    return {
        "status": "exploratory_known_truth_comparison",
        "pythia_trained_models": False,
        "external_knn_baseline_fitted": True,
        "seed": seed, "n_entities": n_entities,
        "repetitions_per_design": repetitions,
        "losses": {"pinball": {"quantile": .9}, "window_miss": {"window": .1}},
        "summary": summarize(rows),
        "by_loss": {
            loss: summarize([row for row in rows if row["loss"] == loss])
            for loss in ("pinball", "window_miss")
        },
        "by_design": {
            name: summarize([row for row in rows if row["design"] == name])
            for name in DESIGN_NAMES
        },
    }
