"""Aggregate-only retrospective Mayo quantile and within-window risk audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from horos.represent.grouping import nearest_group_hierarchy
from horos.represent.pairing import global_nearest_pairs
from pythia.core.decision_order import monotone_window_floor_closure
from pythia.core.mae_family_envelope import (
    decision_pair_group_envelope, geometry_routed_decision_envelope,
)

from .mayo_replicate import CONTINUOUS_RANGES, NOTE_RANGES, ONE_HOT_PREFIXES, _scaled_columns


MAYO_DECISIONS = (
    ("q10_pinball", "pinball", .1, None),
    ("q90_pinball", "pinball", .9, None),
    ("w15_miss", "window_miss", None, 15.),
    ("w30_miss", "window_miss", None, 30.),
    ("w60_miss", "window_miss", None, 60.),
)


def run_mayo_extended_decision_losses(
    derived_path: str | Path, *, n_periods: int = 4,
    lipschitz_bound: float = 300.,
    group_witness_rule: str = "radial",
    group_pinball_range_rule: str = "asymptotic",
    window_policy: str = "legacy",
) -> dict:
    """Evaluate five losses on identical earliest-patient chronological panels.

    The OOF MAE-selected model is held fixed, not selected separately for
    quantile or window losses. No patient row, prediction, or note is returned.
    """
    if n_periods < 2 or not np.isfinite(lipschitz_bound) or lipschitz_bound < 0:
        raise ValueError("valid periods and nonnegative illustrative cap required")
    if window_policy not in {"legacy", "safe_group", "geometry_route"}:
        raise ValueError("window_policy must be legacy, safe_group, or geometry_route")
    path = Path(derived_path)
    frame = pd.read_parquet(path / "df.parquet")
    features = pd.read_parquet(path / "XB.parquet")
    predictions = np.load(path / "PB.npy")
    y_all = frame["case_actual_duration_time"].to_numpy(float)
    winner = int(np.argmin(np.mean(np.abs(predictions - y_all[None, :]), axis=1)))
    ordered = (
        frame.assign(_row=np.arange(len(frame)))
        .sort_values("scheduled_in_room")
        .drop_duplicates("patient_durable_id", keep="first")
        .sort_values("scheduled_in_room")["_row"].to_numpy()
    )
    periods = np.array_split(ordered, n_periods)
    alpha_per_cell = .05 / (n_periods * 2 * len(MAYO_DECISIONS))
    output = []
    for period_id, selected in enumerate(periods, start=1):
        indices = selected[: 2 * (len(selected) // 2)]
        if len(indices) < 32:
            raise ValueError("too few distinct patients in a chronological panel")
        y = y_all[indices]
        if not np.isfinite(y).all() or np.any((y < 0) | (y > 900)):
            raise ValueError("observed duration violates illustrative [0,900] cap")
        frozen = predictions[winner, indices]
        entities = frame["patient_durable_id"].astype(str).to_numpy()[indices]
        tabular_cols = _scaled_columns(features, indices, CONTINUOUS_RANGES)
        for name in features.columns:
            if name.startswith(ONE_HOT_PREFIXES):
                values = features[name].to_numpy(float)[indices]
                values[~np.isfinite(values)] = 0.
                tabular_cols.append(np.clip(values, 0, 1))
        tabular = np.column_stack(tabular_cols)
        note_values = np.column_stack([
            tabular, *_scaled_columns(features, indices, NOTE_RANGES),
        ])
        representations = {}
        for label, x in (
            ("tabular_metric", tabular),
            ("tabular_plus_note_values_and_applicability", note_values),
        ):
            pairs = global_nearest_pairs(x, entities)
            groups = nearest_group_hierarchy(x, entities, max_group_size=16)
            losses = {}
            for loss_label, loss, quantile, window in MAYO_DECISIONS:
                geometry_ratio = float(
                    lipschitz_bound * np.median(pairs.distances)
                    / (900. if loss == "pinball" else window)
                )
                if window_policy == "geometry_route":
                    result = geometry_routed_decision_envelope(
                        x, y, entities, pairs.left_indices, pairs.right_indices,
                        groups, loss=loss, quantile=quantile, window=window,
                        lipschitz_bound=lipschitz_bound,
                        outcome_range_bound=900. if loss == "pinball" else None,
                        alpha=alpha_per_cell,
                    )
                else:
                    window_route = (
                        window_policy == "safe_group" and loss == "window_miss"
                    )
                    result = decision_pair_group_envelope(
                        x, y, entities, pairs.left_indices, pairs.right_indices,
                        groups, loss=loss, quantile=quantile, window=window,
                        lipschitz_bound=lipschitz_bound,
                        outcome_range_bound=900. if loss == "pinball" else None,
                        alpha=alpha_per_cell,
                        group_witness_rule=(
                            "exact_lipschitz" if window_route else group_witness_rule
                        ),
                        group_pinball_range_rule=group_pinball_range_rule,
                        family_policy="grouped" if window_route else "adaptive",
                        methods=(
                            ("conditional_bentkus", "conditional_entity_mcd")
                            if window_route else (
                                "conditional_hoeffding",
                                "conditional_empirical_bernstein",
                                "conditional_bentkus",
                            )
                        ),
                    )
                if result["status"] != "complete":
                    losses[loss_label] = {
                        "status": "refused", "reason": result["reason"],
                    }
                    continue
                residual = y - frozen
                if loss == "pinball":
                    achieved = float(np.mean(np.maximum(
                        quantile * residual, (quantile - 1) * residual,
                    )))
                else:
                    achieved = float(np.mean(np.abs(residual) > window))
                row = {
                    "status": "exploratory", "claim_tier": "exploratory",
                    "bound": result["bound"], "selected_family": result["selected_family"],
                    "paired_bound": result["family_bounds"].get("paired"),
                    "grouped_bound": result["family_bounds"].get("grouped"),
                    "geometry_ratio": geometry_ratio,
                    "achieved_frozen_oof_loss": achieved,
                    "lower_bound_le_achieved_check": result["bound"] <= achieved + 1e-12,
                }
                if loss == "window_miss":
                    row["attainable_within_window_percentage_upper"] = result[
                        "attainable_within_window_percentage_upper"
                    ]
                    row["achieved_within_window_percentage"] = 100 * (1 - achieved)
                losses[loss_label] = row
            if all(losses[f"w{width}_miss"]["status"] == "exploratory"
                   for width in (15, 30, 60)):
                closed = monotone_window_floor_closure({
                    float(width): losses[f"w{width}_miss"]["bound"]
                    for width in (15, 30, 60)
                })
                for width, bound in closed.items():
                    cell = losses[f"w{int(width)}_miss"]
                    previous = cell["bound"]
                    cell["bound_pre_window_order_closure"] = previous
                    cell["bound"] = bound
                    cell["window_order_closure_changed"] = bound > previous + 1e-12
                    if cell["window_order_closure_changed"]:
                        cell["selected_family"] = "monotone_window_closure"
                    cell["attainable_within_window_percentage_upper"] = 100 * (1 - bound)
                    cell["lower_bound_le_achieved_check"] = (
                        bound <= cell["achieved_frozen_oof_loss"] + 1e-12
                    )
            for probability in (.75, .9):
                supported_windows = [
                    width for width in (15., 30., 60.)
                    if losses[f"w{int(width)}_miss"].get("bound", 0.)
                    > 1. - probability
                ]
                losses[f"absolute_error_p{int(100 * probability)}_strict_floor_minutes"] = (
                    max(supported_windows) if supported_windows else None
                )
            representations[label] = losses
        output.append({
            "period": period_id, "n_patients": int(len(indices)),
            "representations": representations,
        })
    total = sum(row["n_patients"] for row in output)
    weighted_summary = {}
    for representation_name in (
        "tabular_metric", "tabular_plus_note_values_and_applicability",
    ):
        losses = {}
        for loss_label, _, _, _ in MAYO_DECISIONS:
            cells = [row["representations"][representation_name][loss_label]
                     for row in output]
            if any(cell["status"] != "exploratory" for cell in cells):
                losses[loss_label] = {"status": "refused"}
                continue
            bound = float(sum(row["n_patients"] * cell["bound"]
                              for row, cell in zip(output, cells)) / total)
            achieved = float(sum(row["n_patients"] * cell["achieved_frozen_oof_loss"]
                                 for row, cell in zip(output, cells)) / total)
            summary = {"bound": bound, "achieved_frozen_oof_loss": achieved,
                       "lower_bound_le_achieved_check": bound <= achieved + 1e-12}
            if loss_label.startswith("w"):
                summary["attainable_within_window_percentage_upper"] = 100 * (1 - bound)
                summary["achieved_within_window_percentage"] = 100 * (1 - achieved)
            losses[loss_label] = summary
        for probability in (.75, .9):
            supported = [width for width in (15., 30., 60.)
                         if losses[f"w{int(width)}_miss"].get("bound", 0.)
                         > 1. - probability]
            losses[f"absolute_error_p{int(100 * probability)}_strict_floor_minutes"] = (
                max(supported) if supported else None
            )
        weighted_summary[representation_name] = losses
    return {
        "status": "exploratory_retrospective_multi_loss",
        "claim_tier": "exploratory",
        "n_unique_patients": int(len(ordered)),
        "n_used_patients": int(sum(row["n_patients"] for row in output)),
        "loss_registry": [item[0] for item in MAYO_DECISIONS],
        "lipschitz_bound_illustrative_not_audited": lipschitz_bound,
        "outcome_range_cap_illustrative_not_population_proved": 900.,
        "alpha_per_loss_representation_period_cell": alpha_per_cell,
        "simultaneous_confidence_if_all_assumptions_hold": .95,
        "simultaneous_window_order_closure": True,
        "pythia_trained_predictive_model": False,
        "group_witness_rule": group_witness_rule,
        "group_pinball_range_rule": group_pinball_range_rule,
        "window_policy": window_policy,
        "reason_not_certified": (
            "retrospective OOF winner selected for MAE; quantile and coverage-action "
            "smoothness, patient independence, outcome-range population cap, and "
            "feature timing are not independently audited"
        ),
        "weighted_summary": weighted_summary,
        "periods": output,
    }
