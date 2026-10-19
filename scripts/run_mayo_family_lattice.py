#!/usr/bin/env python3
"""Aggregate-only retrospective Mayo representation-lattice diagnostic.

No predictor is trained and no note, identifier, row-level label, or row-level
endpoint is written.  The statistical output is exploratory: one earliest
case per patient removes repeated entities, but temporal exchangeability and a
population support audit are not established by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from pythia import (
    finite_cell_information_lattice,
    finite_refinement_permutation_test,
)


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


def _tokens(text):
    return TOKEN.findall(str(text).lower())


def _canonical_tokens(tokens):
    return [token for token in tokens
            if not (len(token) >= 3 and set(token) == {"x"})]


def _metadata_profile(note_count, char_count):
    return f"n{_fixed_bin(note_count, (0, 1, 2, 4))}|c{_fixed_bin(char_count, (0, 1000, 4000, 16000))}"


def _bag_profile(tokens, metadata):
    clean = _canonical_tokens(tokens)
    buckets = np.zeros(4, int)
    numeric = 0
    for token in clean:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        buckets[digest[0] % 4] += 1
        numeric += token.isdigit()
    dominant = int(np.argmax(buckets)) if clean else 0
    return (f"{metadata}|t{_fixed_bin(len(clean), (0, 64, 256, 1024))}"
            f"|d{_fixed_bin(numeric, (0, 2, 8, 32))}|h{dominant}")


def _permutation_within(labels, rng):
    result = np.arange(len(labels))
    for value in np.unique(labels):
        index = np.flatnonzero(labels == value)
        result[index] = rng.permutation(index)
    return result


def _node_summary(result, name):
    raw = result.node_results[name]
    closed = result.order_result.intervals[name]
    return {
        "n_cells": raw.n_cells,
        "singleton_fraction": raw.singleton_fraction,
        "effective_cell_size": raw.effective_cell_size,
        "raw_lower": raw.bound,
        "closed_lower": closed.lower,
        "saa_floor_estimate": raw.empirical_saa_score,
        "ordered_floor_estimate": closed.estimate,
        "concentration_width": raw.concentration_width,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabular", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    columns = [
        "case_durable_id", "patient_durable_id", "scheduled_in_room",
        "case_actual_duration_time", "primary_procedure_name",
    ]
    frame = pd.read_parquet(args.tabular, columns=columns)
    frame["_scheduled"] = pd.to_datetime(frame["scheduled_in_room"], errors="coerce")
    frame = (frame.sort_values(["_scheduled", "case_durable_id"])
             .drop_duplicates("patient_durable_id").reset_index(drop=True))
    notes = (pd.read_parquet(args.notes, columns=[
        "case_durable_id", "note_text", "n_pre_cutoff_notes_used", "note_chars"])
             .drop_duplicates("case_durable_id").set_index("case_durable_id"))
    aligned = notes.reindex(frame["case_durable_id"])
    texts = aligned["note_text"].fillna("").astype(str).to_numpy()
    note_count = aligned["n_pre_cutoff_notes_used"].fillna(0).to_numpy(int)
    char_count = aligned["note_chars"].fillna(0).to_numpy(int)
    y = frame["case_actual_duration_time"].to_numpy(float)
    procedure = frame["primary_procedure_name"].fillna("missing").astype(str).to_numpy()
    family = np.asarray([_family(value) for value in procedure])
    token_rows = [_tokens(text) for text in texts]
    metadata = np.asarray([_metadata_profile(n, c) for n, c in zip(note_count, char_count)])
    bag = np.asarray([_bag_profile(tokens, meta)
                      for tokens, meta in zip(token_rows, metadata)])

    rng = np.random.default_rng(args.seed)
    within_index = _permutation_within(procedure, rng)
    global_index = rng.permutation(len(frame))
    profiles = {
        "intact_meta": metadata,
        "intact_bag": bag,
        "within_meta": metadata[within_index],
        "within_bag": bag[within_index],
        "global_meta": metadata[global_index],
        "global_bag": bag[global_index],
    }
    representations = {
        "global": np.repeat("all", len(frame)),
        "family": family,
        "procedure": procedure,
    }
    for name, profile in profiles.items():
        representations[name] = np.asarray([
            f"{proc}|{value}" for proc, value in zip(procedure, profile)])
    edges = (
        ("family", "global"), ("procedure", "family"),
        ("intact_meta", "procedure"), ("intact_bag", "intact_meta"),
        ("within_meta", "procedure"), ("within_bag", "within_meta"),
        ("global_meta", "procedure"), ("global_bag", "global_meta"),
    )
    result = finite_cell_information_lattice(
        representations, y, np.arange(len(y)), richer_to_poorer=edges,
        loss="mae", outcome_range_bound=900., alpha=.05,
        representation_provenance={name: f"mayo-{name}-profile-v1"
                                   for name in representations},
        support_provenance="retrospective-clinical-[0,900]-working-cap",
        nesting_provenance="deterministic-procedure-family-and-profile-map-v1",
        iid_rows_attested=False, outcome_blind_attested=True,
        nesting_attested=True, exploratory_sampling_override=True)

    # Bag profiles are functions of token multisets, so a token permutation is
    # an exact code-level invariance.  Recompute after actual random shuffles.
    shuffled_equal = True
    x_suffix_equal = True
    for row, (tokens, meta, original, target) in enumerate(
            zip(token_rows, metadata, bag, y)):
        shuffled = list(tokens)
        np.random.default_rng(args.seed + row).shuffle(shuffled)
        shuffled_equal &= _bag_profile(shuffled, meta) == original
        target_x = list(tokens) + ["x" * max(3, int(round(target)))]
        x_suffix_equal &= _bag_profile(target_x, meta) == original

    leaked = finite_cell_information_lattice(
        {"target_x_count": np.asarray([f"x-{int(round(value))}" for value in y])},
        y, np.arange(len(y)), richer_to_poorer=(), loss="mae",
        outcome_range_bound=900., alpha=.05,
        representation_provenance={"target_x_count": "duration-coded-x-channel"},
        support_provenance="retrospective-clinical-[0,900]-working-cap",
        nesting_provenance="single-node-leakage-falsification",
        iid_rows_attested=True, outcome_blind_attested=False,
        nesting_attested=True)

    # Exploratory conditional signal tests.  Two frozen ladder increments are
    # tested and Bonferroni-adjusted; no predictor is fitted.
    metadata_signal = finite_refinement_permutation_test(
        procedure, metadata, y, loss="mae", permutations=args.permutations,
        seed=args.seed + 100, registry_frozen_before_outcomes=True,
        outcome_blind_attested=True, conditional_exchangeability_attested=False,
        exploratory_exchangeability_override=True,
        provenance={"registry": "mayo-fixed-note-metadata-profile-v1"})
    intact_meta_label = representations["intact_meta"]
    bag_signal = finite_refinement_permutation_test(
        intact_meta_label, bag, y, loss="mae", permutations=args.permutations,
        seed=args.seed + 101, registry_frozen_before_outcomes=True,
        outcome_blind_attested=True, conditional_exchangeability_attested=False,
        exploratory_exchangeability_override=True,
        provenance={"registry": "mayo-fixed-hashed-bag-profile-v1"})

    def signal_summary(test):
        return {
            "claim_tier": test.claim_tier,
            "coarse_saa_score": test.coarse_saa_score,
            "refined_saa_score": test.refined_saa_score,
            "observed_gain": test.observed_gain,
            "p_value": test.p_value,
            "bonferroni_p_two_tests": min(1., 2. * test.p_value),
            "null_mean": test.null_mean,
            "null_quantiles": dict(test.null_quantiles),
            "permutations": test.permutations,
            "interpretation": (
                "valid conditional Monte Carlo p-value" if test.claim_tier == "conditional"
                else "exploratory Monte Carlo tail area; exchangeability not attested"),
        }

    payload = {
        "status": "complete", "study_tier": "exploratory",
        "lattice_claim_tier": result.claim_tier,
        "n_unique_patients": len(frame),
        "n_with_pre_cutoff_notes": int(np.sum(note_count > 0)),
        "family_alpha": result.alpha, "spent_alpha": result.spent_alpha,
        "model_training_performed": result.diagnostics["model_training_performed"],
        "nodes": {name: _node_summary(result, name) for name in representations},
        "conditional_signal_tests": {
            "note_metadata_beyond_procedure": signal_summary(metadata_signal),
            "hashed_bag_beyond_procedure_and_metadata": signal_summary(bag_signal),
        },
        "metamorphic_checks": {
            "token_shuffle_bag_profile_exact_invariance": bool(shuffled_equal),
            "target_coded_x_removed_by_canonical_parser_exact_invariance": bool(x_suffix_equal),
            "raw_target_coded_x_status": leaked.status,
            "raw_target_coded_x_reason": leaked.refusal["reason"],
        },
        "limitations": [
            "retrospective Mayo development data; no known Bayes-risk gold",
            "one earliest row per patient does not prove iid sampling or temporal stability",
            "[0,900] is a working clinical cap, not an independently audited population support",
            "fixed hashed bag profiles are diagnostics, not trained or clinically validated NLP",
            "permutation branches test behavior and do not establish a Blackwell relation to intact notes",
            "no positive value-of-notes claim follows from lower endpoints alone",
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
