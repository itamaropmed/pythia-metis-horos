"""Aggregate-only temporal stability audit of model-free Mayo pair bounds.

This is a retrospective sensitivity analysis, not an assumption certificate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from horos.represent.pairing import global_nearest_pairs
from pythia.core.replicates import (
    paired_replicate_clipped_envelope, paired_replicate_lower_bound,
)

from .mayo_replicate import CONTINUOUS_RANGES, NOTE_RANGES, ONE_HOT_PREFIXES, _scaled_columns


def _period_pair_bounds(
    matrix: np.ndarray, y: np.ndarray, entities: np.ndarray,
    *, alpha: float, lipschitz_grid: tuple[float, ...],
    outcome_term_clips: tuple[float, ...] = (56.25, 112.5, 225.0, 450.0),
) -> dict:
    design = global_nearest_pairs(matrix, entities)
    kwargs = dict(
        y_left=y[design.left_indices], y_right=y[design.right_indices],
        distances=design.distances,
        left_entities=entities[design.left_indices],
        right_entities=entities[design.right_indices],
        pair_blocks=design.pair_blocks, loss="mae", distance_bound=1.0,
        outcome_difference_bound=900.0, alpha=alpha,
    )
    comparison = {}
    for concentration in ("conditional_hoeffding", "conditional_empirical_bernstein"):
        comparison[concentration] = [
            {
                "L": float(L),
                "bound": result["bound"],
                "corrected_center": result["estimate"],
                "radius": result["radius"],
            }
            for L in lipschitz_grid
            for result in [paired_replicate_lower_bound(
                **kwargs, lipschitz_bound=L, concentration=concentration,
            )]
        ]
    comparison["conditional_empirical_bernstein_clipped_envelope"] = [
        {
            "L": float(L), "bound": result["bound"],
            "corrected_center": result["estimate"],
            "radius": result["radius"],
            "selected_clip": result["selected_clip"],
        }
        for L in lipschitz_grid
        for result in [paired_replicate_clipped_envelope(
            **kwargs, lipschitz_bound=L,
            outcome_term_clips=outcome_term_clips,
            concentration="conditional_empirical_bernstein",
        )]
    ]
    return {
        "n_pairs": int(design.left_indices.size),
        "mean_pair_distance": float(np.mean(design.distances)),
        "comparison": comparison,
        "empirical_bernstein_minus_hoeffding": [
            float(eb["bound"] - hf["bound"])
            for eb, hf in zip(
                comparison["conditional_empirical_bernstein"],
                comparison["conditional_hoeffding"],
            )
        ],
        "clipped_minus_unclipped_empirical_bernstein": [
            float(clipped["bound"] - plain["bound"])
            for clipped, plain in zip(
                comparison["conditional_empirical_bernstein_clipped_envelope"],
                comparison["conditional_empirical_bernstein"],
            )
        ],
    }


def run_mayo_temporal_stability(
    derived_path: str | Path,
    *, n_periods: int = 4,
    lipschitz_grid: tuple[float, ...] = (0.0, 100.0, 300.0, 600.0),
) -> dict:
    """Compare matched concentration methods on disjoint chronological panels."""
    if n_periods < 2:
        raise ValueError("temporal audit needs at least two periods")
    if not lipschitz_grid or any(value < 0 for value in lipschitz_grid):
        raise ValueError("L grid must be nonempty and nonnegative")
    path = Path(derived_path)
    frame = pd.read_parquet(path / "df.parquet")
    features = pd.read_parquet(path / "XB.parquet")
    predictions = np.load(path / "PB.npy")
    full_y = frame["case_actual_duration_time"].to_numpy(float)
    oof_winner = int(np.argmin(np.mean(np.abs(predictions - full_y[None, :]), axis=1)))
    earliest = (
        frame.assign(_row=np.arange(len(frame)))
        .sort_values("scheduled_in_room")
        .drop_duplicates("patient_durable_id", keep="first")
    )
    ordered = earliest.sort_values("scheduled_in_room")["_row"].to_numpy()
    periods = np.array_split(ordered, n_periods)
    specification = {
        "continuous_ranges": CONTINUOUS_RANGES,
        "note_ranges": NOTE_RANGES,
        "one_hot_prefixes": ONE_HOT_PREFIXES,
        "missing_value": 0.5, "metric": "mean_l1",
        "patient_row": "earliest scheduled case",
    }
    specification_hash = hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode()
    ).hexdigest()
    result = {
        "status": "exploratory_retrospective_temporal_stability",
        "claim_tier": "exploratory", "pythia_trained_models": False,
        "reason_not_certified": (
            "calendar-period patient independence, pre-outcome feature timing, "
            "and Mayo smoothness caps have not been independently audited"
        ),
        "metric_specification_hash": specification_hash,
        "n_unique_patients": int(ordered.size),
        "n_periods": n_periods,
        "alpha_allocation": (
            "Bonferroni over three concentration methods, two representations, "
            "and all periods, plus "
            "the clip grid within the clipped method; each conditional "
            "outcome-term event covers the full L grid"
        ),
        "alpha_per_method_period_representation": float(0.05 / (6 * n_periods)),
        "outcome_term_clips": [56.25, 112.5, 225.0, 450.0],
        "periods": [],
    }
    for period_id, period_indices in enumerate(periods, start=1):
        # An outcome-blind last-row drop gives an even number for perfect matching.
        indices = period_indices[: (len(period_indices) // 2) * 2]
        if len(indices) < 4:
            raise ValueError("a temporal period has too few patients")
        y = full_y[indices]
        entities = frame["patient_durable_id"].astype(str).to_numpy()[indices]
        tabular_columns = _scaled_columns(features, indices, CONTINUOUS_RANGES)
        for name in features.columns:
            if name.startswith(ONE_HOT_PREFIXES):
                values = features[name].to_numpy(float)[indices]
                values[~np.isfinite(values)] = 0.0
                tabular_columns.append(np.clip(values, 0.0, 1.0))
        tabular = np.column_stack(tabular_columns)
        notes = np.column_stack([
            tabular, *_scaled_columns(features, indices, NOTE_RANGES)
        ])
        date_values = pd.to_datetime(frame["scheduled_in_room"].iloc[indices])
        result["periods"].append({
            "period": period_id,
            "first_month": date_values.min().strftime("%Y-%m"),
            "last_month": date_values.max().strftime("%Y-%m"),
            "n_patients": int(len(indices)),
            "frozen_full_cohort_oof_winner_mae": float(np.mean(np.abs(
                predictions[oof_winner, indices] - y
            ))),
            "representations": {
                "tabular_metric": _period_pair_bounds(
                    tabular, y, entities,
                    alpha=0.05 / (6 * n_periods),
                    lipschitz_grid=lipschitz_grid,
                ),
                "tabular_plus_note_values_and_applicability": _period_pair_bounds(
                    notes, y, entities,
                    alpha=0.05 / (6 * n_periods),
                    lipschitz_grid=lipschitz_grid,
                ),
            },
        })
    return result
