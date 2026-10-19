"""Retrospective Mayo diagnostic for the model-free replicate bound."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from horos.represent.pairing import (
    clustered_nearest_pairs, global_nearest_pairs, randomized_block_pairs,
)
from pythia import PairMatrix
from pythia.core.replicates import (
    paired_replicate_ensemble_sensitivity, paired_replicate_sensitivity_curve,
)


CONTINUOUS_RANGES = {
    "patient_age": (0.0, 120.0), "bmi": (0.0, 100.0),
    "case_num_procedures": (0.0, 10.0), "case_num_panels": (0.0, 5.0),
    "case_num_providers": (0.0, 25.0), "scheduled_room_minutes": (0.0, 900.0),
    "hist_sp_median": (0.0, 900.0), "hist_proc_median": (0.0, 900.0),
    "hist_surgeon_median": (0.0, 900.0), "hist_patient_prior_mean": (0.0, 900.0),
}
NOTE_RANGES = {
    "llm_mapping_complexity": (1.0, 5.0),
    "llm_procedural_complexity_5": (1.0, 5.0),
    "llm_anatomical_complexity": (1.0, 5.0),
    "llm_n_complexity_drivers": (0.0, 6.0),
    "llm_arrhythmia_complexity": (1.0, 5.0),
    "llm_comorbidity_burden": (1.0, 5.0),
    "llm_epicardial_access_anticipated": (0.0, 1.0),
    "llm_obstructive_sleep_apnea": (0.0, 1.0),
    "llm_refractory_af_markers": (0.0, 1.0),
    "llm_tricuspid_regurgitation_severity": (0.0, 3.0),
    "llm_vt_morphologies_count": (0.0, 13.0),
}
ONE_HOT_PREFIXES = (
    "comorbid_", "case_urgency_level_", "case_surgery_patient_class_",
    "case_anesthesia_", "primary_procedure_name_",
)


def _scaled_columns(frame: pd.DataFrame, indices: np.ndarray, ranges: dict) -> list[np.ndarray]:
    columns = []
    for name, (lower, upper) in ranges.items():
        values = (frame[name].to_numpy(float)[indices] - lower) / (upper - lower)
        missing = ~np.isfinite(values)
        values[missing] = 0.5
        columns.extend([np.clip(values, 0.0, 1.0), missing.astype(float)])
    return columns


def run_mayo_replicate_diagnostic(
    derived_path: str | Path,
    *,
    seeds: tuple[int, ...] = tuple(range(20260915, 20260925)),
    block_size: int = 8,
    lipschitz_grid: tuple[float, ...] = (0, 25, 50, 100, 200, 300, 400, 500, 600),
) -> dict:
    """Return aggregate diagnostics only; no patient-level artifact is emitted."""
    path = Path(derived_path)
    frame = pd.read_parquet(path / "df.parquet")
    features = pd.read_parquet(path / "XB.parquet")
    predictions = np.load(path / "PB.npy")
    indices = (
        frame.assign(_index=np.arange(len(frame)))
        .sort_values("scheduled_in_room")
        .drop_duplicates("patient_durable_id", keep="first")["_index"]
        .to_numpy()
    )
    n_eligible = int(indices.size)
    indices = indices[: (indices.size // block_size) * block_size]
    y = frame["case_actual_duration_time"].to_numpy(float)[indices]
    entities = frame["patient_durable_id"].astype(str).to_numpy()[indices]
    week_clusters = pd.to_datetime(frame["scheduled_in_room"]).dt.strftime("%G-W%V").to_numpy()[indices]
    tabular = _scaled_columns(features, indices, CONTINUOUS_RANGES)
    for name in features.columns:
        if name.startswith(ONE_HOT_PREFIXES):
            values = features[name].to_numpy(float)[indices]
            values[~np.isfinite(values)] = 0.0
            tabular.append(np.clip(values, 0.0, 1.0))
    tabular_matrix = np.column_stack(tabular)
    note_columns = _scaled_columns(features, indices, NOTE_RANGES)
    representations = {
        "tabular_metric": tabular_matrix,
        "tabular_plus_note_values_and_applicability": np.column_stack(
            [tabular_matrix, *note_columns]
        ),
    }
    specification = {
        "continuous_ranges": CONTINUOUS_RANGES, "note_ranges": NOTE_RANGES,
        "one_hot_prefixes": ONE_HOT_PREFIXES, "missing_value": 0.5,
        "metric": "mean_l1", "patient_row": "earliest scheduled case",
    }
    specification_hash = hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode()
    ).hexdigest()
    result = {
        "status": "exploratory", "claim_tier": "exploratory",
        "reason_not_certified": (
            "retrospective historical cohort; iid population sampling, conditional entity "
            "independence, and the Lipschitz cap have not been externally audited"
        ),
        "pythia_trained_models": False, "n_patients": n_eligible,
        "n_used_patients": int(indices.size), "seed_registry": list(seeds),
        "best_oof_mae_on_selected_rows": float(
            np.min(np.mean(np.abs(predictions[:, indices] - y[None, :]), axis=1))
        ),
        "metric_specification_hash": specification_hash, "representations": {},
    }
    for name, matrix in representations.items():
        pair_designs = [
            randomized_block_pairs(matrix, entities, block_size=block_size, seed=seed)
            for seed in seeds
        ]
        matrices = tuple(
            PairMatrix(
                y[design.left_indices], y[design.right_indices], design.distances,
                entities[design.left_indices], entities[design.right_indices],
                design.pair_blocks, meta=design.meta,
            )
            for design in pair_designs
        )
        sensitivity = paired_replicate_ensemble_sensitivity(
            matrices, loss="mae",
            lipschitz_bounds=np.asarray(lipschitz_grid), distance_bound=1.0,
            outcome_difference_bound=900.0,
        )
        conditional_sensitivity = paired_replicate_ensemble_sensitivity(
            matrices, loss="mae", lipschitz_bounds=np.asarray(lipschitz_grid),
            distance_bound=1.0, outcome_difference_bound=900.0,
            concentration="conditional_hoeffding",
        )
        global_design = global_nearest_pairs(matrix, entities)
        global_sensitivity = paired_replicate_sensitivity_curve(
            y[global_design.left_indices], y[global_design.right_indices],
            global_design.distances, entities[global_design.left_indices],
            entities[global_design.right_indices], global_design.pair_blocks,
            loss="mae", lipschitz_bounds=np.asarray(lipschitz_grid),
            distance_bound=1.0, outcome_difference_bound=900.0,
            concentration="conditional_empirical_bernstein",
        )
        weekly_design = clustered_nearest_pairs(matrix, entities, week_clusters)
        weekly_sensitivity = paired_replicate_sensitivity_curve(
            y[weekly_design.left_indices], y[weekly_design.right_indices],
            weekly_design.distances, entities[weekly_design.left_indices],
            entities[weekly_design.right_indices], weekly_design.pair_blocks,
            loss="mae", lipschitz_bounds=np.asarray(lipschitz_grid),
            distance_bound=1.0, outcome_difference_bound=900.0,
            concentration="conditional_block_empirical_bernstein",
        )
        result["representations"][name] = {
            "n_features": int(matrix.shape[1]),
            "mean_pair_distance_range": [
                float(min(np.mean(design.distances) for design in pair_designs)),
                float(max(np.mean(design.distances) for design in pair_designs)),
            ],
            "design_registry": [dict(design.meta) for design in pair_designs],
            "curve_confidence": sensitivity["confidence"],
            "curve_simultaneous": sensitivity["simultaneous"],
            "selection_safe": sensitivity["selection_safe"],
            "curve": sensitivity["curve"],
            "conditional_fixed_design_curve": conditional_sensitivity["curve"],
            "global_fixed_design_empirical_bernstein_curve": global_sensitivity["curve"],
            "global_mean_pair_distance": float(np.mean(global_design.distances)),
            "global_design": dict(global_design.meta),
            "weekly_cluster_fixed_design_curve": weekly_sensitivity["curve"],
            "weekly_cluster_mean_pair_distance": float(np.mean(weekly_design.distances)),
            "weekly_cluster_design": dict(weekly_design.meta),
            "weekly_cluster_warning": (
                "calendar-week construction is outcome-blind, but cross-week independence "
                "and the choice of week as the dependence unit are not established"
            ),
            "conditional_curve_estimand": (
                "average conditional Bayes MAE over the audited patient covariates"
            ),
            "curve_comparison_warning": (
                "the population and fixed-design curves have different estimands and their "
                "separate 95% events are not a joint selection guarantee"
            ),
        }
    return result
