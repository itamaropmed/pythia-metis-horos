#!/usr/bin/env python3
"""Aggregate-only Mayo finite-cell diagnostic retaining repeated patients."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from pythia import clustered_block_saa_portfolio, clustered_finite_cell_saa_lower_bound


TOKEN = re.compile(r"[a-z0-9]+")
FAMILY_PATTERNS = (
    ("extraction", ("extract", "removal", "explant")),
    ("revision", ("revision", "generator", "upgrade", "pocket")),
    ("af_pvi", ("af ", "fibrillation", "pvi", "pulmonary vein", "pfa")),
    ("vt_pvc", ("vt", "ventricular tach", "pvc")),
    ("other_ablation", ("ablat",)),
    ("implant", ("implant", "insert", "pacemaker", "icd", "crt", "leadless",
                 "loop recorder")),
    ("diagnostic", ("study", "test", "diagnostic", "tilt", "cardiovers")),
)


def _family(value):
    text = str(value).lower()
    for name, patterns in FAMILY_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return name
    return "other"


def _fixed_bin(value, cuts):
    return int(np.searchsorted(np.asarray(cuts, float), float(value), side="right"))


def _canonical_tokens(text):
    return [token for token in TOKEN.findall(str(text).lower())
            if not (len(token) >= 3 and set(token) == {"x"})]


def _metadata_profile(note_count, char_count):
    return (f"n{_fixed_bin(note_count, (0, 1, 2, 4))}"
            f"|c{_fixed_bin(char_count, (0, 1000, 4000, 16000))}")


def _bag_profile(tokens, metadata):
    buckets = np.zeros(4, int)
    numeric = 0
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        buckets[digest[0] % 4] += 1
        numeric += token.isdigit()
    dominant = int(np.argmax(buckets)) if tokens else 0
    return (f"{metadata}|t{_fixed_bin(len(tokens), (0, 64, 256, 1024))}"
            f"|d{_fixed_bin(numeric, (0, 2, 8, 32))}|h{dominant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabular", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    columns = ["case_durable_id", "patient_durable_id", "scheduled_in_room",
               "case_actual_duration_time", "primary_procedure_name"]
    frame = pd.read_parquet(args.tabular, columns=columns)
    frame = frame.dropna(subset=["patient_durable_id", "case_actual_duration_time"])
    frame["_scheduled"] = pd.to_datetime(frame["scheduled_in_room"], errors="coerce")
    frame = frame.sort_values(["_scheduled", "case_durable_id"]).reset_index(drop=True)
    notes = (pd.read_parquet(args.notes, columns=[
        "case_durable_id", "note_text", "n_pre_cutoff_notes_used", "note_chars"])
        .drop_duplicates("case_durable_id").set_index("case_durable_id"))
    aligned = notes.reindex(frame["case_durable_id"])
    text = aligned["note_text"].fillna("").astype(str).to_numpy()
    count = aligned["n_pre_cutoff_notes_used"].fillna(0).to_numpy(int)
    chars = aligned["note_chars"].fillna(0).to_numpy(int)
    y = frame["case_actual_duration_time"].to_numpy(float)
    patients = frame["patient_durable_id"].astype(str).to_numpy()
    procedure = frame["primary_procedure_name"].fillna("missing").astype(str).to_numpy()
    family = np.asarray([_family(value) for value in procedure])
    token_rows = [_canonical_tokens(value) for value in text]
    metadata = np.asarray([_metadata_profile(n, c) for n, c in zip(count, chars)])
    bag = np.asarray([_bag_profile(tokens, meta)
                      for tokens, meta in zip(token_rows, metadata)])
    representations = {
        "global": np.repeat("all", len(frame)),
        "family": family,
        "procedure": procedure,
        "procedure_note_metadata": np.asarray([
            f"{p}|{m}" for p, m in zip(procedure, metadata)]),
        "procedure_hashed_bag": np.asarray([
            f"{p}|{b}" for p, b in zip(procedure, bag)]),
    }
    alpha_per_node = .05 / len(representations)
    nodes = {}
    for name, labels in representations.items():
        result = clustered_finite_cell_saa_lower_bound(
            labels, y, patients, loss="mae", outcome_range_bound=900.,
            alpha=alpha_per_node, representation_provenance=f"mayo-{name}-v1",
            support_provenance="retrospective-clinical-[0,900]-working-cap",
            independent_clusters_attested=False, outcome_blind_attested=True,
            exploratory_cluster_override=True)
        support_free = clustered_block_saa_portfolio(
            labels, y, patients, loss="mae",
            block_cluster_sizes=(32, 64, 128, 256, 512),
            score_caps=(5., 10., 20., 40., 80., 160.),
            alpha=alpha_per_node, partition_seed=args.seed,
            representation_provenance=f"mayo-{name}-v1",
            partition_provenance="frozen-seeded-patient-blocks-v1",
            independent_clusters_attested=False,
            outcome_blind_labels_attested=True,
            partition_frozen_before_outcomes=True,
            exploratory_cluster_override=True)
        values, sizes = np.unique(labels, return_counts=True)
        del values
        invalid_row_width = 900. * np.sqrt(np.log(1 / alpha_per_node) / (2 * len(y)))
        nodes[name] = {
            "claim_tier": result.claim_tier, "n_cells": result.n_cells,
            "singleton_fraction": float(np.mean(sizes == 1)),
            "empirical_saa_score": result.empirical_saa_score,
            "cluster_lower": result.bound,
            "cluster_width": result.concentration_width,
            "support_free_block_lower": support_free.bound,
            "support_free_selected_route": support_free.selected_route,
            "support_free_selected_score_cap": support_free.selected_score_cap,
            "support_free_alpha_per_candidate": support_free.alpha_per_candidate,
            "invalid_row_iid_width_negative_control": float(invalid_row_width),
            "invalid_row_iid_lower_negative_control": float(max(
                0., result.empirical_saa_score - invalid_row_width)),
        }

    shuffled_equal = True
    x_removed_equal = True
    for row, (tokens, meta, original, target) in enumerate(zip(token_rows, metadata, bag, y)):
        shuffled = list(tokens)
        np.random.default_rng(args.seed + row).shuffle(shuffled)
        shuffled_equal &= _bag_profile(shuffled, meta) == original
        x_tokens = list(tokens) + ["x" * max(3, int(round(target)))]
        x_removed_equal &= _bag_profile(
            [token for token in x_tokens
             if not (len(token) >= 3 and set(token) == {"x"})], meta) == original
    leaked = clustered_finite_cell_saa_lower_bound(
        np.asarray([f"x-{int(round(value))}" for value in y]), y, patients,
        loss="mae", outcome_range_bound=900., alpha=.05,
        representation_provenance="duration-coded-x-channel",
        support_provenance="retrospective-clinical-[0,900]-working-cap",
        independent_clusters_attested=False, outcome_blind_attested=False,
        exploratory_cluster_override=True)

    _, cluster_sizes = np.unique(patients, return_counts=True)
    payload = {
        "status": "complete", "study_tier": "exploratory",
        "n_cases": len(frame), "n_patients": len(cluster_sizes),
        "n_repeated_cases": int(len(frame) - len(cluster_sizes)),
        "maximum_cases_per_patient": int(cluster_sizes.max()),
        "effective_cluster_count": float(len(frame) ** 2 / np.sum(cluster_sizes ** 2)),
        "alpha_family": .05, "alpha_per_node": alpha_per_node,
        "nodes": nodes,
        "metamorphic_checks": {
            "token_shuffle_bag_profile_exact_invariance": bool(shuffled_equal),
            "target_coded_x_removed_exact_invariance": bool(x_removed_equal),
            "raw_target_coded_x_status": leaked.status,
            "raw_target_coded_x_reason": leaked.reason,
        },
        "limitations": [
            "cluster independence and temporal stability are not attested",
            "[0,900] is a working cap rather than an independently audited population support",
            "row-IID values are labeled invalid negative controls and are not certificates",
            "hashed text profiles are fixed diagnostics rather than trained NLP models",
            "support-free block routes are a separate five-node family and cannot be max-selected with support-based routes without a joint alpha ledger",
            "no patient, case, note, label, or row-level result is written",
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
