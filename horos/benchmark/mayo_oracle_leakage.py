"""Aggregate-only, deliberately invalid post-outcome feature stress test.

No modified note text or patient row is written. This is a falsification check,
not a route to a retrospective or prospective certificate.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from horos.represent.pairing import global_nearest_pairs
from horos.benchmark.mayo_replicate import (
    CONTINUOUS_RANGES, ONE_HOT_PREFIXES, _scaled_columns,
)
from pythia import PairMatrix, ReplicateGuardSpec, Representation, TaskCard, certify
from pythia.core.replicates import paired_replicate_lower_bound


_ORACLE = re.compile(r"\[PYTHIA-ORACLE\] actual duration: (\d+(?:\.\d+)?) minutes")


def _inject_and_extract(text: str, duration: float) -> float:
    """Exercise literal note augmentation and parsing without persisting text."""
    augmented = f"{text}\n[PYTHIA-ORACLE] actual duration: {duration:.6f} minutes"
    matches = _ORACLE.findall(augmented)
    if len(matches) != 1:
        raise ValueError("oracle marker was not uniquely recoverable")
    return float(matches[0])


def run_mayo_oracle_leakage(
    derived_path: str | Path,
    *,
    n_patients: int = 2048,
    seed: int = 20260912,
    fractions: tuple[float, ...] = (0.1, 0.5, 1.0),
) -> dict:
    """Test 10%, 50%, and full in-memory duration injection into notes/tabular."""
    path = Path(derived_path)
    frame = pd.read_parquet(path / "df.parquet")
    notes = pd.read_parquet(
        path / "note_bundles_recut.parquet", columns=["case_durable_id", "note_text"]
    )
    predictions = np.load(path / "PB.npy")
    earliest = (
        frame.assign(_row=np.arange(len(frame)))
        .sort_values("scheduled_in_room")
        .drop_duplicates("patient_durable_id", keep="first")
    )
    eligible = earliest[earliest["case_durable_id"].isin(notes["case_durable_id"])]["_row"].to_numpy()
    if n_patients < 4 or n_patients > eligible.size or n_patients % 2:
        raise ValueError("n_patients must be even, at least four, and no larger than eligibility")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(eligible, size=n_patients, replace=False))
    y = frame["case_actual_duration_time"].to_numpy(float)[indices]
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 900)):
        raise ValueError("duration outside the predeclared [0, 900] diagnostic range")
    cases = frame["case_durable_id"].to_numpy()[indices]
    entities = frame["patient_durable_id"].astype(str).to_numpy()[indices]
    note_lookup = notes.set_index("case_durable_id")["note_text"]
    texts = note_lookup.loc[cases].fillna("").astype(str).to_numpy()
    # Reuse the frozen full-cohort OOF winner; never choose a predictor using
    # the diagnostic subsample's outcomes.
    full_y = frame["case_actual_duration_time"].to_numpy(float)
    baseline_index = int(np.argmin(np.mean(np.abs(predictions - full_y[None, :]), axis=1)))
    baseline = predictions[baseline_index, indices]
    baseline_mae = float(np.mean(np.abs(baseline - y)))
    features = pd.read_parquet(path / "XB.parquet")
    tabular_columns = _scaled_columns(features, indices, CONTINUOUS_RANGES)
    for name in features.columns:
        if name.startswith(ONE_HOT_PREFIXES):
            values = features[name].to_numpy(float)[indices]
            values[~np.isfinite(values)] = 0.0
            tabular_columns.append(np.clip(values, 0.0, 1.0))
    tabular_metric = np.column_stack(tabular_columns)
    scheduled_column = 2 * list(CONTINUOUS_RANGES).index("scheduled_room_minutes")
    duplicate_metric = np.column_stack([
        tabular_metric,
        np.repeat(tabular_metric[:, scheduled_column, None], 20, axis=1),
    ])
    duplicate_control = {}
    for metric_name, metric in (
        ("original", tabular_metric),
        ("twenty_extra_schedule_copies", duplicate_metric),
    ):
        design = global_nearest_pairs(metric, entities)
        diagnostic = paired_replicate_lower_bound(
            y[design.left_indices], y[design.right_indices], design.distances,
            entities[design.left_indices], entities[design.right_indices],
            design.pair_blocks, loss="mae", lipschitz_bound=300.0,
            distance_bound=1.0, outcome_difference_bound=900.0,
            concentration="conditional_empirical_bernstein",
        )
        duplicate_control[metric_name] = {
            "n_features": int(metric.shape[1]),
            "mean_pair_distance": diagnostic["mean_distance"],
            "pseudo_bound_at_unaudited_L_300": diagnostic["bound"],
            "pseudo_center_at_unaudited_L_300": diagnostic["estimate"],
        }
    order = rng.permutation(n_patients)
    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("injection fractions must lie in (0, 1]")
        selected = np.zeros(n_patients, dtype=bool)
        selected[order[:int(round(fraction * n_patients))]] = True
        tabular_oracle = np.full(n_patients, np.nan)
        note_oracle = np.full(n_patients, np.nan)
        tabular_oracle[selected] = y[selected]
        for i in np.flatnonzero(selected):
            note_oracle[i] = _inject_and_extract(texts[i], y[i])
        if not np.array_equal(np.isnan(tabular_oracle), np.isnan(note_oracle)):
            raise AssertionError("note and tabular injection coverage diverged")
        if not np.allclose(tabular_oracle[selected], note_oracle[selected], atol=1e-6):
            raise AssertionError("the injected note duration did not round-trip")
        mixed = baseline.copy()
        mixed[selected] = note_oracle[selected]

        # The exact target is a coordinate in [0, 1]. For a fully injected
        # sample, its conditional median is 900-Lipschitz in this coordinate;
        # every paired score is exactly zero. An invalid zero-smoothness cap
        # yields a misleading positive pseudo-bound.
        oracle_bound = None
        invalid_zero_smoothness_bound = None
        invalid_zero_smoothness_center = None
        if selected.all():
            design = global_nearest_pairs((y / 900.0)[:, None], entities)
            kwargs = dict(
                y_left=y[design.left_indices], y_right=y[design.right_indices],
                distances=design.distances,
                left_entities=entities[design.left_indices],
                right_entities=entities[design.right_indices],
                pair_blocks=design.pair_blocks, loss="mae",
                distance_bound=1.0, outcome_difference_bound=900.0,
                concentration="conditional_hoeffding",
            )
            oracle_bound = paired_replicate_lower_bound(**kwargs, lipschitz_bound=900.0)
            invalid_zero_smoothness = paired_replicate_lower_bound(
                **kwargs, lipschitz_bound=0.0
            )
            invalid_zero_smoothness_bound = invalid_zero_smoothness["bound"]
            invalid_zero_smoothness_center = invalid_zero_smoothness["raw_estimate"]
            pair_matrix = PairMatrix(
                kwargs["y_left"], kwargs["y_right"], kwargs["distances"],
                kwargs["left_entities"], kwargs["right_entities"],
                kwargs["pair_blocks"], meta={
                    "pairing_outcome_blind": False,
                    "conditionally_independent_entities": True,
                },
            )
            refusal = certify(
                TaskCard("case_actual_duration_time", "mae"), pair_matrix,
                Representation(
                    "duration-from-note", provenance={
                        "uses_target": True, "available_at_prediction_time": False,
                    },
                ),
                ReplicateGuardSpec(
                    "oracle-test", 900.0, 1.0, 900.0,
                    "post-outcome falsification only", "oracle-test",
                    concentration="conditional_hoeffding",
                ),
            )
            if refusal.status != "refused":
                raise AssertionError("Pythia accepted a known leaked representation")
        rows.append({
            "fraction_requested": fraction,
            "n_injected": int(selected.sum()),
            "oracle_prediction_mae_on_injected": float(np.mean(np.abs(note_oracle[selected] - y[selected]))),
            "mixed_oracle_plus_frozen_oof_mae": float(np.mean(np.abs(mixed - y))),
            "random_mask_expected_mixed_mae": float(
                baseline_mae * (1.0 - selected.mean())
            ),
            "full_oracle_pair_bound_with_L_900": None if oracle_bound is None else oracle_bound["bound"],
            "full_oracle_pair_estimate_with_L_900": None if oracle_bound is None else oracle_bound["estimate"],
            "full_oracle_pseudo_bound_with_invalid_L_0": invalid_zero_smoothness_bound,
            "full_oracle_pseudo_center_with_invalid_L_0": invalid_zero_smoothness_center,
            "public_api_status": None if oracle_bound is None else refusal.status,
            "public_api_reason": None if oracle_bound is None else refusal.refusal["reason"],
        })
    return {
        "status": "exploratory_falsification_only",
        "deployable": False, "claim_tier": "refused",
        "reason": "the injected duration is the post-outcome target itself",
        "pythia_trained_models": False,
        "n_patients": n_patients, "seed": seed,
        "baseline_frozen_oof_mae_on_selected_patients": baseline_mae,
        "deterministic_feature_duplication_control": {
            "true_information_change": "none; copied coordinate is a deterministic function of existing tabular features",
            "claim_tier": "exploratory", "smoothness_cap_audited": False,
            "metrics": duplicate_control,
        },
        "arms": rows,
    }
