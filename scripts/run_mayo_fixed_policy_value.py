#!/usr/bin/env python3
"""Aggregate-only retrospective audit of a saved Mayo held-out policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pythia import fixed_policy_information_value_lower_bound


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.predictions)
    prediction_column = "pred_" + metadata["selected_variant"]
    y = frame["case_actual_duration_time"].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    losses = np.abs(y - prediction)
    expected = float(metadata["final_metrics"]["heldout_mae"])
    if len(frame) != int(metadata["heldout_cases"]):
        raise AssertionError("held-out row count does not match metadata")
    if not np.isclose(np.mean(losses), expected, rtol=0, atol=1e-12):
        raise AssertionError("saved held-out MAE does not match metadata")

    summaries = {}
    for name, labels in (
            ("procedure_family", frame["procedure_family"].astype(str).to_numpy()),
            ("procedure", frame["primary_procedure_name"].astype(str).to_numpy())):
        result = fixed_policy_information_value_lower_bound(
            labels, y, frame["case_durable_id"].to_numpy(), losses,
            loss="mae", loss_range_bound=900., outcome_range_bound=900.,
            alpha=.05, upper_methods=("dKelly",),
            representation_provenance=f"heldout-{name}-v1",
            support_provenance="retrospective-[0,900]-working-cap",
            policy_provenance="31.5564-oof-selected-heldout-policy",
            iid_rows_attested=False, outcome_blind_coarse_attested=True,
            policy_frozen_before_evaluation=True,
            policy_refinement_measurable_attested=True,
            policy_preoutcome_attested=True,
            exploratory_sampling_override=True)
        summaries[name] = {
            "claim_tier": result.claim_tier,
            "n_cells": result.coarse_result.n_cells,
            "coarse_saa_score": result.coarse_result.empirical_saa_score,
            "coarse_risk_lower": result.coarse_risk_lower,
            "policy_mae": result.policy_risk_estimate,
            "policy_risk_upper": result.policy_risk_upper,
            "raw_value_endpoint": result.raw_value_endpoint,
            "value_lower_bound": result.value_lower_bound,
        }
    payload = {
        "status": "complete", "study_tier": "exploratory",
        "heldout_rows": len(frame), "patient_overlap_with_training": 0,
        "verified_heldout_mae": float(np.mean(losses)),
        "maximum_observed_absolute_loss": float(np.max(losses)),
        "selected_variant": metadata["selected_variant"],
        "metadata_sha256": _sha256(args.metadata),
        "predictions_sha256": _sha256(args.predictions),
        "results": summaries,
        "limitations": [
            "the temporal holdout has been viewed during retrospective development",
            "iid sampling and temporal stability are not attested",
            "[0,900] is a working cap rather than an independently audited population support",
            "the endpoint concerns value beyond procedure/family, not notes alone",
            "zero lower value does not prove that the refined representation has zero value",
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
