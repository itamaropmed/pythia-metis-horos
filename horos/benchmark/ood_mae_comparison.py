"""Ten-family OOD known-risk test for train-free MAE bounds and estimators."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from horos.represent.grouping import nearest_group_hierarchy
from horos.represent.pairing import global_nearest_pairs
from pythia.core.group_witness import grouped_decision_envelope
from pythia.core.mae_family_envelope import mae_pair_group_envelope
from pythia.core.replicates import paired_replicate_clipped_envelope

from .dependence_comparison import _wilson_lower
from .ood_mae_designs import OOD_DESIGNS, ood_mae_design
from .omniscient_lp_baseline import jackknife_lipschitz_mae_lp


CLIPS = (.05, .1, .2, .4, .8, 1.6)


def _local_mae_jackknife(y: np.ndarray, groups: np.ndarray) -> float:
    """No-predictor-fit exploratory local Bayes-score bias correction."""
    scores = []
    for rows in groups:
        values = y[rows]
        mid = len(values) // 2
        full = float(np.mean(np.abs(values - np.median(values))))
        half = .5 * sum(float(np.mean(np.abs(part - np.median(part))))
                          for part in (values[:mid], values[mid:]))
        scores.append(2 * full - half)
    return float(max(0, np.mean(scores)))


def run_ood_mae_comparison(
    *, seed: int = 20260925, repetitions: int = 10, n_entities: int = 1600,
    design_names: tuple[str, ...] = OOD_DESIGNS,
    design_fn=ood_mae_design,
) -> dict:
    """Compare equal-confidence paired/grouped bounds and point estimators."""
    if repetitions < 1 or n_entities < 32 or n_entities % 16:
        raise ValueError("positive repetitions and n_entities divisible by 16 >=32 required")
    if not design_names or len(set(design_names)) != len(design_names):
        raise ValueError("nonempty unique design registry required")
    seeds = np.random.SeedSequence(seed).spawn(len(design_names) * repetitions)
    rows = []
    cursor = 0
    for name in design_names:
        for _ in range(repetitions):
            draw_id = cursor
            rng = np.random.default_rng(seeds[cursor]); cursor += 1
            x, y, l, conditional_truth = design_fn(rng, name, n_entities)
            truth = float(np.mean(conditional_truth))
            entity = np.arange(n_entities)
            pairs = global_nearest_pairs(x, entity)
            args = (
                y[pairs.left_indices], y[pairs.right_indices], pairs.distances,
                entity[pairs.left_indices], entity[pairs.right_indices], pairs.pair_blocks,
            )
            paired_rows = [paired_replicate_clipped_envelope(
                *args, loss="mae", lipschitz_bound=l, distance_bound=1.,
                outcome_difference_bound=None, outcome_term_clips=CLIPS,
                alpha=.025, concentration=method,
            ) for method in (
                "conditional_hoeffding", "conditional_empirical_bernstein",
            )]
            if any(row["status"] != "complete" for row in paired_rows):
                raise RuntimeError("known-risk paired design was refused")
            paired = max(paired_rows, key=lambda row: row["bound"])
            hierarchy = nearest_group_hierarchy(x, entity, max_group_size=16)
            grouped = grouped_decision_envelope(
                x, y, entity, hierarchy, loss="mae", lipschitz_bound=l,
                outcome_range_bound=None, outcome_term_clips=CLIPS,
                alpha=.05,
            )
            if grouped["status"] != "complete":
                raise RuntimeError("known-risk grouped design was refused")
            combined = mae_pair_group_envelope(
                x, y, entity, pairs.left_indices, pairs.right_indices,
                hierarchy, lipschitz_bound=l, outcome_term_clips=CLIPS,
                alpha=.05,
            )
            if combined["status"] != "complete":
                raise RuntimeError("known-risk combined design was refused")
            _, neighbors = cKDTree(x).query(x, k=min(52, n_entities))
            knn_predict = np.median(y[neighbors[:, 1:]], axis=1)
            row = {
                "design": name, "draw_id": draw_id, "truth": truth,
                "paired_95": paired["bound"],
                "grouped_95": grouped["bound"],
                "combined_95": combined["bound"],
                "grouped_center": grouped["estimate"],
                "local_jackknife": _local_mae_jackknife(y, hierarchy[max(hierarchy)]),
                "knn_oof": float(np.mean(np.abs(y - knn_predict))),
            }
            if n_entities <= 250:
                lp = jackknife_lipschitz_mae_lp(
                    x, y, lipschitz_bound=l, split_seed=43,
                )
                if lp["status"] != "complete":
                    raise RuntimeError(lp["reason"])
                row["lp_jackknife"] = lp["estimate"]
                row["omniscient_lp_raw"] = lp["full_lp"]
            rows.append(row)

    def summarize(selected: list[dict]) -> dict:
        truth = np.asarray([row["truth"] for row in selected])
        output = {
            "total_independent_draws": len(selected),
            "median_analytic_bayes_mae": float(np.median(truth)),
        }
        for method in ("paired_95", "grouped_95", "combined_95"):
            values = np.asarray([row[method] for row in selected])
            valid = values <= truth + 1e-12
            output[method] = {
                "valid": int(valid.sum()), "total": len(selected),
                "wilson_lower_95": _wilson_lower(int(valid.sum()), len(selected)),
                "median_bound": float(np.median(values)),
                "median_usable_tightness": float(np.median(
                    np.where(valid, values / truth, 0)
                )),
            }
        estimators = ["grouped_center", "local_jackknife", "knn_oof"]
        if "lp_jackknife" in selected[0]:
            estimators.extend(("lp_jackknife", "omniscient_lp_raw"))
        for method in estimators:
            values = np.asarray([row[method] for row in selected])
            output[method] = {
                "median_absolute_error": float(np.median(np.abs(values - truth))),
            }
        pair = np.asarray([row["paired_95"] for row in selected])
        group = np.asarray([row["grouped_95"] for row in selected])
        combined = np.asarray([row["combined_95"] for row in selected])
        jack = np.abs(np.asarray([row["local_jackknife"] for row in selected]) - truth)
        knn = np.abs(np.asarray([row["knn_oof"] for row in selected]) - truth)
        output["paired_comparisons"] = {
            "grouped_bound_strict_wins": int(np.sum(group > pair + 1e-12)),
            "paired_bound_strict_wins": int(np.sum(pair > group + 1e-12)),
            "jackknife_estimator_strict_wins": int(np.sum(jack + 1e-12 < knn)),
            "knn_estimator_strict_wins": int(np.sum(knn + 1e-12 < jack)),
            "median_grouped_minus_paired_bound": float(np.median(group - pair)),
            "combined_bound_strict_wins_over_paired": int(np.sum(combined > pair + 1e-12)),
            "paired_bound_strict_wins_over_combined": int(np.sum(pair > combined + 1e-12)),
            "median_combined_minus_paired_bound": float(np.median(combined - pair)),
        }
        if "lp_jackknife" in selected[0]:
            lp_error = np.abs(np.asarray([row["lp_jackknife"] for row in selected]) - truth)
            output["paired_comparisons"].update({
                "lp_jackknife_estimator_strict_wins_over_knn": int(np.sum(lp_error + 1e-12 < knn)),
                "knn_strict_wins_over_lp_jackknife": int(np.sum(knn + 1e-12 < lp_error)),
                "lp_jackknife_strict_wins_over_raw_lp": int(np.sum(
                    lp_error + 1e-12 < np.abs(np.asarray([
                        row["omniscient_lp_raw"] for row in selected
                    ]) - truth)
                )),
            })
        return output

    return {
        "status": "exploratory_ood_known_truth",
        "seed": seed, "n_entities": n_entities,
        "repetitions_per_design": repetitions,
        "design_names": list(design_names),
        "outcome_term_clip_registry": list(CLIPS),
        "pythia_trained_predictive_model": False,
        "external_knn_predictor_fitted": True,
        "bound_estimand": "average conditional Bayes MAE at fixed audited covariates",
        "summary": summarize(rows),
        "by_design": {
            name: summarize([row for row in rows if row["design"] == name])
            for name in design_names
        },
    }


def run_second_ood_mae_comparison(
    *, seed: int = 20260930, repetitions: int = 20, n_entities: int = 512,
) -> dict:
    """Run the distinct predeclared challenge without changing estimators."""
    from .ood_mae_designs_second import OOD_DESIGNS_SECOND, ood_mae_design_second

    return run_ood_mae_comparison(
        seed=seed, repetitions=repetitions, n_entities=n_entities,
        design_names=OOD_DESIGNS_SECOND, design_fn=ood_mae_design_second,
    )
