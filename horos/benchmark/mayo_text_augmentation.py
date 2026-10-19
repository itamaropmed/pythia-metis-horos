"""Aggregate-only, train-free Mayo note-perturbation diagnostic.

This is a retrospective representation stress test, not a released
certificate. No prediction model or text vocabulary is fitted, and the
returned payload contains neither text nor patient/case-level records.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from horos.represent.grouping import nearest_group_hierarchy
from horos.represent.pairing import global_nearest_pairs
from horos.represent.perturb import TextPerturbationSpec, perturb_texts
from pythia import geometry_routed_decision_envelope

from .mayo_replicate import CONTINUOUS_RANGES, ONE_HOT_PREFIXES, _scaled_columns


CONTROLS = (
    TextPerturbationSpec("identity", 0.0, 20260914),
    TextPerturbationSpec("case_dropout", 0.5, 20260914),
    TextPerturbationSpec("case_dropout", 1.0, 20260914),
    TextPerturbationSpec("character_shuffle", 1.0, 20260914),
    TextPerturbationSpec("token_shuffle", 1.0, 20260914),
    TextPerturbationSpec("span_mask", 0.5, 20260914),
    TextPerturbationSpec("segment_dropout", 0.5, 20260914),
    TextPerturbationSpec("segment_duplicate", 0.5, 20260914),
)


def _hash_text(texts: np.ndarray) -> np.ndarray:
    """Stateless map into [0,1]; fixed across every control."""
    vectorizer = HashingVectorizer(
        n_features=64, analyzer="word", ngram_range=(1, 2),
        alternate_sign=False, binary=False, norm="l2", lowercase=True,
    )
    return vectorizer.transform(texts.tolist()).toarray().astype(float)


def evaluate_text_controls(
    tabular: np.ndarray,
    texts: np.ndarray,
    outcomes: np.ndarray,
    entities: np.ndarray,
    row_ids: np.ndarray,
    *,
    controls: tuple[TextPerturbationSpec, ...] = CONTROLS,
    lipschitz_bound: float = 300.0,
    alpha: float = .05,
) -> dict:
    """Evaluate fixed, label-free perturbations on identical entity rows."""
    x = np.asarray(tabular, float)
    y = np.asarray(outcomes, float).ravel()
    entity = np.asarray(entities).astype(str).ravel()
    ids = np.asarray(row_ids).ravel()
    notes = np.asarray(texts, dtype=object).ravel()
    n = len(y)
    if (x.ndim != 2 or x.shape[0] != n or len(entity) != n or len(ids) != n
            or len(notes) != n or n < 4 or n % 2 or len(np.unique(entity)) != n):
        raise ValueError("even cohort with aligned, unique sampling entities required")
    if not controls or len({(s.kind, s.fraction, s.seed) for s in controls}) != len(controls):
        raise ValueError("nonempty unique perturbation registry required")
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("tabular metric must be finite and in [0,1]")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 900)):
        raise ValueError("illustrative [0,900] outcome span violated")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    rows = {}
    for spec in controls:
        altered, audit = perturb_texts(notes, ids, spec)
        representation = np.column_stack((x, _hash_text(altered)))
        pairs = global_nearest_pairs(representation, entity)
        hierarchy = nearest_group_hierarchy(representation, entity, max_group_size=8)
        key = "identity" if spec.kind == "identity" else f"{spec.kind}_{spec.fraction:g}"
        losses = {}
        for loss, parameter in (("pinball_q90", {"loss": "pinball", "quantile": .9,
                                                  "outcome_range_bound": 900.0}),
                                ("w30_miss", {"loss": "window_miss", "window": 30.0})):
            result = geometry_routed_decision_envelope(
                representation, y, entity, pairs.left_indices, pairs.right_indices,
                hierarchy, lipschitz_bound=lipschitz_bound,
                alpha=alpha / (2 * len(controls)), **parameter,
            )
            losses[loss] = {
                "status": result["status"], "claim_tier": "exploratory",
                "bound": result.get("bound"),
                "selected_family": result.get("selected_family"),
                "geometry_ratio": result.get("geometry_ratio"),
                "reason": result.get("reason"),
            }
        rows[key] = {
            "changed_rows": audit.changed_rows,
            "empty_after": audit.empty_after,
            "characters_after": audit.characters_after,
            "source_hash": audit.source_hash,
            "result_hash": audit.result_hash,
            "mean_pair_distance": float(np.mean(pairs.distances)),
            "losses": losses,
        }
    return {
        "status": "exploratory", "claim_tier": "exploratory",
        "pythia_trained_models": False,
        "text_transform_fitted": False,
        "n_entities": n,
        "n_with_text": int(np.sum(notes != "")),
        "alpha_across_controls_and_losses": alpha,
        "metric": "mean_l1_of_fixed_tabular_and_64_stateless_hashed_text_coordinates",
        "lipschitz_bound": lipschitz_bound,
        "outcome_range": [0, 900],
        "limitations": [
            "Mayo retrospective outcomes previously influenced geometry-route thresholds",
            "the Lipschitz cap and 900-minute population support are not externally audited",
            "a changed representation changes the metric, so bound monotonicity is not implied",
            "note provenance and conditional entity independence need external audit",
            "no known Bayes gold exists for Mayo; a larger bound is not automatically better",
        ],
        "controls": rows,
    }


def run_mayo_text_augmentation(derived_path: str | Path, *, n_entities: int = 256) -> dict:
    """Read local private artifacts and return only aggregate diagnostics."""
    if n_entities < 4 or n_entities % 2:
        raise ValueError("n_entities must be even and at least four")
    path = Path(derived_path)
    frame = pd.read_parquet(path / "df.parquet")
    features = pd.read_parquet(path / "XB.parquet")
    bundle = pd.read_parquet(path / "note_bundles_recut.parquet")
    bundle = bundle.drop_duplicates("case_durable_id").set_index("case_durable_id")
    indices = (
        frame.assign(_row=np.arange(len(frame)))
        .sort_values("scheduled_in_room")
        .drop_duplicates("patient_durable_id", keep="first")
        .sort_values("scheduled_in_room")["_row"].to_numpy()[:n_entities]
    )
    if len(indices) != n_entities:
        raise ValueError("not enough distinct eligible patients")
    tabular_columns = _scaled_columns(features, indices, CONTINUOUS_RANGES)
    for name in features.columns:
        if name.startswith(ONE_HOT_PREFIXES):
            values = features[name].to_numpy(float)[indices]
            values[~np.isfinite(values)] = 0.0
            tabular_columns.append(np.clip(values, 0.0, 1.0))
    case_ids = frame["case_durable_id"].to_numpy()[indices]
    texts = bundle["note_text"].reindex(case_ids).fillna("").astype(str).to_numpy()
    result = evaluate_text_controls(
        np.column_stack(tabular_columns), texts,
        frame["case_actual_duration_time"].to_numpy(float)[indices],
        frame["patient_durable_id"].astype(str).to_numpy()[indices], case_ids,
    )
    result["cohort_rule"] = "earliest chronological case per patient, first n entities"
    result["source_artifact"] = "note_bundles_recut.parquet (private; not exported)"
    return result
