"""Synthetic signing tests for the Metis trusted calibration-artifact boundary."""
from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest
import json

from metis.calibration_evidence import (
    _array_sha256,
    fit_verified_case_risk_map,
    load_verified_calibration_evidence,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed_fixture(tmp_path):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("OpenSSL is required for Ed25519 signature tests")
    private_key = tmp_path / "reviewer-private.pem"
    public_key = tmp_path / "reviewer-public.pem"
    subprocess.run([openssl, "genpkey", "-algorithm", "ED25519", "-out", str(private_key)],
                   check=True, capture_output=True)
    subprocess.run([openssl, "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)],
                   check=True, capture_output=True)

    rng = np.random.default_rng(422)
    n = 500
    y = rng.normal(150., 30., n)
    predictions = {"model_a": y + rng.normal(0., 10., n),
                   "model_b": y + rng.normal(0., 20., n)}
    entity_ids = np.arange(n).astype(str)
    fold_ids = (np.arange(n) % 5).astype(str)
    strata = np.where(np.arange(n) % 2, "complex", "routine")
    artifact = tmp_path / "calibration.npz"
    np.savez_compressed(artifact, y=y, entity_ids=entity_ids, fold_ids=fold_ids,
                        strata=strata,
                        **{f"prediction__{name}": value for name, value in predictions.items()})

    bound_artifacts = {
        "feature_source": tmp_path / "features.bin",
        "feature_pipeline": tmp_path / "feature_pipeline.py",
        "model_a": tmp_path / "model_a.bin",
        "model_b": tmp_path / "model_b.bin",
    }
    bound_artifacts["feature_source"].write_bytes(b"synthetic source feature snapshot")
    bound_artifacts["feature_pipeline"].write_text("# reviewed synthetic outcome-blind pipeline\n")
    bound_artifacts["model_a"].write_bytes(b"frozen model A artifact")
    bound_artifacts["model_b"].write_bytes(b"frozen model B artifact")

    manifest = {
        "schema_version": 1,
        "purpose": "metis-risk-calibration",
        "artifact_sha256": _sha256(artifact),
        "predictions_are_out_of_fold": True,
        "models_frozen_before_risk_calibration": True,
        "feature_pipeline_outcome_blind": True,
        "entity_independence_reviewed": True,
        "split_protocol": "patient_grouped_oof_v1",
        "prediction_time_rule": "scheduled_in_room_minus_2_days",
        "feature_source_sha256": _sha256(bound_artifacts["feature_source"]),
        "feature_pipeline_sha256": _sha256(bound_artifacts["feature_pipeline"]),
        "fold_assignment_sha256": _array_sha256(fold_ids),
        "entity_assignment_sha256": _array_sha256(entity_ids),
        "entity_independence_basis": "one row per unique synthetic entity",
        "model_names": ["model_a", "model_b"],
        "model_sha256": {"model_a": _sha256(bound_artifacts["model_a"]),
                         "model_b": _sha256(bound_artifacts["model_b"])},
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    payload_path = tmp_path / "payload.json"
    signature_path = tmp_path / "signature.bin"
    payload_path.write_bytes(payload)
    subprocess.run([openssl, "pkeyutl", "-sign", "-inkey", str(private_key), "-rawin",
                    "-in", str(payload_path), "-out", str(signature_path)],
                   check=True, capture_output=True)
    manifest["signature_ed25519"] = base64.b64encode(signature_path.read_bytes()).decode()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return artifact, manifest_path, public_key, _sha256(public_key), y, predictions, bound_artifacts


def test_signed_calibration_artifact_is_verified_and_auditable(tmp_path):
    artifact, manifest, public_key, pin, y, predictions, bound = _signed_fixture(tmp_path)
    evidence = load_verified_calibration_evidence(artifact, manifest, public_key, pin,
                                                 bound_artifacts=bound)
    result = fit_verified_case_risk_map(evidence, pinned_reviewer_key_sha256=pin,
                                        min_entities=20, error_cap=200.)
    assert evidence.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result.calibration_evidence_status == "signature_verified_reviewer_attestation"
    assert result.calibration_manifest_sha256 == evidence.manifest_sha256
    case_predictions = {name: float(values[0]) for name, values in predictions.items()}
    with pytest.raises(ValueError, match="requires matching verified deployment model artifacts"):
        result.score("new-case", case_predictions, id_salt="test")
    verified_models = result.verify_deployed_model_artifacts(
        {name: bound[name] for name in predictions}
    )
    with pytest.raises(AttributeError, match="immutable"):
        verified_models.calibration_manifest_sha256 = "forged"
    serialized = json.loads(result.to_json())
    assert serialized["calibration_manifest_sha256"] == evidence.manifest_sha256
    assert "deployment_model_hashes_match" in result.score(
        "new-case", {name: float(values[0]) for name, values in predictions.items()},
        id_salt="test", verified_model_artifacts=verified_models,
    ).model_risks[0].uncertainty_status
    wrong_model = tmp_path / "same-name-wrong-model.bin"
    wrong_model.write_bytes(b"different model, same public name")
    with pytest.raises(ValueError, match="does not match signed calibration"):
        result.verify_deployed_model_artifacts({
            "model_a": wrong_model, "model_b": bound["model_b"],
        })
    with pytest.raises(ValueError, match="independently pinned trust anchor"):
        load_verified_calibration_evidence(artifact, manifest, public_key, "0" * 64,
                                           bound_artifacts=bound)


def test_signed_artifact_hash_detects_post_signature_mutation(tmp_path):
    artifact, manifest, public_key, pin, _, _, bound = _signed_fixture(tmp_path)
    with np.load(artifact, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["y"] = arrays["y"].copy()
    arrays["y"][0] += 1.
    np.savez_compressed(artifact, **arrays)
    with pytest.raises(ValueError, match="artifact hash"):
        load_verified_calibration_evidence(artifact, manifest, public_key, pin,
                                           bound_artifacts=bound)


def test_signed_manifest_fold_assignment_hash_must_match_npz(tmp_path):
    artifact, manifest_path, public_key, pin, _, _, bound = _signed_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    # An authorized signer may still sign a mistaken fold commitment. The
    # verifier checks consistency with the actual artifact rather than trusting it.
    unsigned = {key: value for key, value in manifest.items() if key != "signature_ed25519"}
    unsigned["fold_assignment_sha256"] = "f" * 64
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    payload_path = tmp_path / "bad-fold-payload.json"
    signature_path = tmp_path / "bad-fold-signature.bin"
    private_key = tmp_path / "reviewer-private.pem"
    payload_path.write_bytes(payload)
    openssl = shutil.which("openssl")
    subprocess.run([openssl, "pkeyutl", "-sign", "-inkey", str(private_key), "-rawin",
                    "-in", str(payload_path), "-out", str(signature_path)],
                   check=True, capture_output=True)
    unsigned["signature_ed25519"] = base64.b64encode(signature_path.read_bytes()).decode()
    manifest_path.write_text(json.dumps(unsigned, sort_keys=True, indent=2))
    with pytest.raises(ValueError, match="fold assignments"):
        load_verified_calibration_evidence(artifact, manifest_path, public_key, pin,
                                           bound_artifacts=bound)


def test_signed_manifest_binds_feature_pipeline_and_model_artifacts(tmp_path):
    artifact, manifest, public_key, pin, _, _, bound = _signed_fixture(tmp_path)
    bound["feature_pipeline"].write_text("# changed after reviewer signature\n")
    with pytest.raises(ValueError, match="dependency hash mismatch: feature_pipeline"):
        load_verified_calibration_evidence(artifact, manifest, public_key, pin,
                                           bound_artifacts=bound)


def test_fit_reloads_signed_data_instead_of_trusting_forged_dataclass_arrays(tmp_path):
    artifact, manifest, public_key, pin, y, predictions, bound = _signed_fixture(tmp_path)
    evidence = load_verified_calibration_evidence(artifact, manifest, public_key, pin,
                                                 bound_artifacts=bound)
    forged = replace(evidence, y=evidence.y + 500.)
    result = fit_verified_case_risk_map(forged, pinned_reviewer_key_sha256=pin,
                                        min_entities=20, error_cap=200.)
    expected = float(np.mean(np.abs(y - predictions["model_a"])))
    assert result.cells["model_a"]["global:all"].expected_absolute_error == pytest.approx(expected)

