"""Small-n ten-design comparison with a published-style MAE-floor heuristic."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from horos.represent.pairing import global_nearest_pairs
from pythia.core.replicates import paired_replicate_lower_bound

from .omniscient_lp_baseline import (
    global_lipschitz_mae_lp, jackknife_lipschitz_mae_lp,
)
from .replicate_adversarial import DESIGN_NAMES, _design


def run_omniscient_lp_comparison(
    *, seed: int = 20260922, repetitions: int = 5, n_entities: int = 200,
) -> dict:
    """Point-error comparison; LP output is not treated as a 95% certificate."""
    if repetitions < 1 or n_entities < 4 or n_entities % 2 or n_entities > 250:
        raise ValueError("positive repetitions and even 4<=n_entities<=250 required")
    seeds = np.random.SeedSequence(seed).spawn(len(DESIGN_NAMES) * repetitions)
    rows = []
    cursor = 0
    for name in DESIGN_NAMES:
        for _ in range(repetitions):
            rng = np.random.default_rng(seeds[cursor]); cursor += 1
            x, y, l, _, cap, conditional = _design(rng, name, n_entities)
            truth = float(np.mean(conditional["mae"]))
            entity = np.arange(n_entities)
            pairs = global_nearest_pairs(x, entity)
            paired = paired_replicate_lower_bound(
                y[pairs.left_indices], y[pairs.right_indices], pairs.distances,
                entity[pairs.left_indices], entity[pairs.right_indices], pairs.pair_blocks,
                loss="mae", lipschitz_bound=l, distance_bound=1.,
                outcome_difference_bound=cap,
                concentration="conditional_empirical_bernstein",
            )
            _, neighbors = cKDTree(x).query(x, k=min(52, n_entities))
            predicted = np.median(y[neighbors[:, 1:]], axis=1)
            knn = float(np.mean(np.abs(y - predicted)))
            lp = global_lipschitz_mae_lp(x, y, lipschitz_bound=l)
            jackknife = jackknife_lipschitz_mae_lp(
                x, y, lipschitz_bound=l, split_seed=43,
            )
            if lp["status"] != "complete":
                raise RuntimeError(lp["reason"])
            if jackknife["status"] != "complete":
                raise RuntimeError(jackknife["reason"])
            rows.append({
                "design": name, "truth": truth,
                "lp": lp["empirical_omniscient_mae"],
                "lp_jackknife": jackknife["estimate"],
                "knn_oof": knn,
                "paired_center": paired["estimate"],
                "paired_95_lower": paired["bound"],
            })

    def summarize(selected: list[dict]) -> dict:
        truth = np.asarray([row["truth"] for row in selected])
        errors = {method: np.abs(np.asarray([row[method] for row in selected]) - truth)
                  for method in ("lp", "lp_jackknife", "knn_oof", "paired_center")}
        return {
            "total": len(selected),
            "median_absolute_error": {
                method: float(np.median(value)) for method, value in errors.items()
            },
            "point_estimator_strict_wins": {
                method: int(np.sum(value + 1e-12 < np.minimum.reduce([
                    other for name, other in errors.items() if name != method
                ]))) for method, value in errors.items()
            },
            "lp_below_truth": int(sum(row["lp"] <= row["truth"] + 1e-12 for row in selected)),
            "paired_lower_below_truth": int(sum(
                row["paired_95_lower"] <= row["truth"] + 1e-12 for row in selected
            )),
        }
    return {
        "status": "exploratory_small_n_published_style_comparison",
        "seed": seed, "n_entities": n_entities,
        "repetitions_per_design": repetitions,
        "pythia_trained_predictive_model": False,
        "external_knn_predictor_fitted": True,
        "lp_reference": "Anderson and Bjarnadottir (2024), PLOS ONE, doi:10.1371/journal.pone.0296904",
        "lp_difference_from_paper": (
            "all pair constraints and oracle-declared L; not their selected distance threshold/link rule"
        ),
        "summary": summarize(rows),
        "by_design": {
            name: summarize([row for row in rows if row["design"] == name])
            for name in DESIGN_NAMES
        },
    }
