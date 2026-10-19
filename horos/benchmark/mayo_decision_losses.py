"""Retrospective, aggregate-only Mayo sensitivity for quantile and W30 losses."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from horos.represent.pairing import global_nearest_pairs
from horos.represent.grouping import nearest_group_hierarchy
from pythia.core.group_witness import grouped_decision_envelope
from pythia.core.replicates import paired_replicate_lower_bound

from .mayo_replicate import CONTINUOUS_RANGES, NOTE_RANGES, ONE_HOT_PREFIXES, _scaled_columns


def run_mayo_decision_losses(
    derived_path: str | Path, *, n_periods: int = 4,
    lipschitz_bound: float = 300.0,
) -> dict:
    """Audit fixed OOF predictions; never re-fit a predictor or expose patient rows."""
    if n_periods < 2 or not np.isfinite(lipschitz_bound) or lipschitz_bound < 0:
        raise ValueError("valid periods and nonnegative finite Lipschitz cap required")
    path = Path(derived_path)
    frame = pd.read_parquet(path / "df.parquet")
    features = pd.read_parquet(path / "XB.parquet")
    predictions = np.load(path / "PB.npy")
    y_all = frame["case_actual_duration_time"].to_numpy(float)
    oof_winner = int(np.argmin(np.mean(np.abs(predictions - y_all[None, :]), axis=1)))
    ordered = (
        frame.assign(_row=np.arange(len(frame)))
        .sort_values("scheduled_in_room")
        .drop_duplicates("patient_durable_id", keep="first")
        .sort_values("scheduled_in_room")["_row"].to_numpy()
    )
    periods = np.array_split(ordered, n_periods)
    rows = []
    for period_id, period_indices in enumerate(periods, start=1):
        indices = period_indices[: 2 * (len(period_indices) // 2)]
        if len(indices) < 4:
            raise ValueError("period has too few patients")
        y = y_all[indices]
        frozen_prediction = predictions[oof_winner, indices]
        entities = frame["patient_durable_id"].astype(str).to_numpy()[indices]
        tabular_columns = _scaled_columns(features, indices, CONTINUOUS_RANGES)
        for name in features.columns:
            if name.startswith(ONE_HOT_PREFIXES):
                values = features[name].to_numpy(float)[indices]
                values[~np.isfinite(values)] = 0.0
                tabular_columns.append(np.clip(values, 0.0, 1.0))
        tabular = np.column_stack(tabular_columns)
        notes = np.column_stack([tabular, *_scaled_columns(features, indices, NOTE_RANGES)])
        representations = {}
        for representation_name, matrix in (
            ("tabular_metric", tabular),
            ("tabular_plus_note_values_and_applicability", notes),
        ):
            pairs = global_nearest_pairs(matrix, entities)
            hierarchy = nearest_group_hierarchy(matrix, entities, max_group_size=16)
            args = (
                y[pairs.left_indices], y[pairs.right_indices], pairs.distances,
                entities[pairs.left_indices], entities[pairs.right_indices], pairs.pair_blocks,
            )
            comparison = {}
            for loss in ("pinball", "w30_miss"):
                event_alpha = .05 / (n_periods * 2 * 2)
                kwargs = dict(
                    loss=loss, lipschitz_bound=lipschitz_bound,
                    distance_bound=1.0,
                    outcome_difference_bound=(900.0 if loss == "pinball" else None),
                    quantile=(.9 if loss == "pinball" else None),
                    alpha=event_alpha / 4,
                )
                comparison[loss] = {
                    method: {
                        "bound": result["bound"], "pair_center": result["estimate"],
                        "radius": result["radius"],
                    }
                    for method in ("conditional_hoeffding", "conditional_empirical_bernstein")
                    for result in [paired_replicate_lower_bound(
                        *args, **kwargs, concentration=method,
                    )]
                }
                grouped = grouped_decision_envelope(
                    matrix, y, entities, hierarchy,
                    loss=("window_miss" if loss == "w30_miss" else loss),
                    lipschitz_bound=lipschitz_bound,
                    outcome_range_bound=(900.0 if loss == "pinball" else None),
                    quantile=(.9 if loss == "pinball" else None),
                    window=(30.0 if loss == "w30_miss" else None),
                    alpha=event_alpha / 2,
                )
                if grouped["status"] != "complete":
                    comparison[loss]["grouped"] = {
                        "status": "refused", "reason": grouped["reason"],
                    }
                else:
                    comparison[loss]["grouped"] = {
                        "status": "complete", "bound": grouped["bound"],
                        "group_center": grouped["estimate"],
                        "radius": grouped["radius"],
                        "selected_group_size": grouped["group_size"],
                        "selected_concentration": grouped["concentration"],
                    }
            representations[representation_name] = {
                "n_pairs": int(len(pairs.distances)),
                "mean_pair_distance": float(np.mean(pairs.distances)),
                "losses": comparison,
            }
        residual = y - frozen_prediction
        rows.append({
            "period": period_id, "n_patients": int(len(indices)),
            "achieved_frozen_oof_pinball_0_9": float(np.mean(
                np.maximum(.9 * residual, -.1 * residual)
            )),
            "achieved_frozen_oof_w30_miss": float(np.mean(np.abs(residual) > 30.0)),
            "representations": representations,
        })
    return {
        "status": "exploratory_retrospective_decision_loss_sensitivity",
        "claim_tier": "exploratory", "pythia_trained_models": False,
        "n_unique_patients": int(len(ordered)),
        "lipschitz_bound": lipschitz_bound,
        "alpha_allocation": (
            "Bonferroni over periods, representations, losses, paired/grouped families, "
            "two concentration methods, and grouped size grid"
        ),
        "reason_not_certified": (
            "Mayo Lipschitz caps, quantile/coverage-center smoothness, patient independence, "
            "and pre-outcome feature timing are not independently established; "
            "the full-cohort OOF winner was selected retrospectively"
        ),
        "periods": rows,
    }
