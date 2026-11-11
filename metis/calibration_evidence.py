"""Verify signed, hash-bound calibration inputs before issuing Metis endpoints.

The signature proves that a configured reviewer signed the manifest and that
the arrays match its artifact hash. It does not prove the reviewer's claims;
the signing key must be pinned outside the artifact and controlled by an
independent audit process.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

from .case_risk import FrozenRiskMap, fit_case_risk_map


REQUIRED_ATTESTATIONS = (
    "predictions_are_out_of_fold",
    "models_frozen_before_risk_calibration",
    "feature_pipeline_outcome_blind",
    "entity_independence_reviewed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedCalibrationEvidence:
    """Calibration arrays loaded only after signed-manifest verification."""

    y: np.ndarray
    predictions: Mapping[str, np.ndarray]
    entity_ids: np.ndarray
    fold_ids: np.ndarray
    strata: np.ndarray | None
    model_sha256: Mapping[str, str]
    artifact_sha256: str
    manifest_sha256: str
    reviewer_key_sha256: str
    split_protocol: str
    prediction_time_rule: str
    artifact_path: str
    manifest_path: str
    reviewer_public_key_path: str
    bound_artifacts: Mapping[str, str]


def load_verified_calibration_evidence(
    artifact_path: str | Path,
    manifest_path: str | Path,
    reviewer_public_key_path: str | Path,
    pinned_reviewer_key_sha256: str,
    *,
    bound_artifacts: Mapping[str, str | Path],
) -> VerifiedCalibrationEvidence:
    """Load a signed NPZ calibration artifact with strict row/fold checks.

    Required arrays are ``y``, ``entity_ids``, ``fold_ids`` and one
    ``prediction__<model-name>`` per manifest model. Optional ``strata`` is
    used by the risk map. Object arrays are refused. The manifest must attest
    to frozen-model, OOF, outcome-blind, and entity-independence review; those
    are reviewer claims, not properties inferred by this loader.
    """
    artifact_path, manifest_path, key_path = map(
        Path, (artifact_path, manifest_path, reviewer_public_key_path)
    )
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise ValueError("calibration artifact must be a regular file")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("calibration manifest must be a regular file")
    if not key_path.is_file() or key_path.is_symlink():
        raise ValueError("reviewer public key must be a regular file")
    key_hash = _sha256(key_path)
    if key_hash != pinned_reviewer_key_sha256:
        raise ValueError("reviewer key does not match the independently pinned trust anchor")

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("calibration manifest must be a JSON object")
    signature_text = manifest.get("signature_ed25519")
    if not isinstance(signature_text, str):
        raise ValueError("calibration manifest requires an Ed25519 signature")
    try:
        signature = base64.b64decode(signature_text, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("calibration signature must be valid base64") from exc
    unsigned = {key: value for key, value in manifest.items() if key != "signature_ed25519"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("OpenSSL is required to verify calibration signatures")
    with tempfile.TemporaryDirectory(prefix="metis-calibration-") as temp:
        payload_path = Path(temp) / "manifest.json"
        signature_path = Path(temp) / "manifest.sig"
        payload_path.write_bytes(payload)
        signature_path.write_bytes(signature)
        result = subprocess.run(
            [openssl, "pkeyutl", "-verify", "-pubin", "-inkey", str(key_path),
             "-rawin", "-in", str(payload_path), "-sigfile", str(signature_path)],
            capture_output=True, check=False,
        )
    if result.returncode:
        raise ValueError("calibration manifest signature verification failed")

    if manifest.get("schema_version") != 1 or manifest.get("purpose") != "metis-risk-calibration":
        raise ValueError("unsupported calibration manifest schema or purpose")
    if manifest.get("artifact_sha256") != _sha256(artifact_path):
        raise ValueError("calibration artifact hash does not match signed manifest")
    for attestation in REQUIRED_ATTESTATIONS:
        if manifest.get(attestation) is not True:
            raise ValueError(f"signed reviewer attestation is missing: {attestation}")
    split_protocol = manifest.get("split_protocol")
    prediction_time_rule = manifest.get("prediction_time_rule")
    if not isinstance(split_protocol, str) or not split_protocol.strip():
        raise ValueError("manifest must name the OOF split protocol")
    if not isinstance(prediction_time_rule, str) or not prediction_time_rule.strip():
        raise ValueError("manifest must state the feature-availability rule")
    for field in ("feature_source_sha256", "feature_pipeline_sha256", "fold_assignment_sha256",
                  "entity_assignment_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"manifest requires a valid {field}")
    if not isinstance(manifest.get("entity_independence_basis"), str) or not manifest["entity_independence_basis"].strip():
        raise ValueError("manifest must explain its independent-entity sampling basis")
    model_hashes = manifest.get("model_sha256")
    model_names = manifest.get("model_names")
    if (not isinstance(model_names, list) or len(model_names) < 2
            or len(set(model_names)) != len(model_names)
            or not all(isinstance(name, str) and name.strip() for name in model_names)):
        raise ValueError("manifest must list at least two unique models")
    if not isinstance(model_hashes, dict) or set(model_hashes) != set(model_names):
        raise ValueError("manifest model hashes must match its model names")
    for value in model_hashes.values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("model_sha256 values must be lowercase SHA-256 digests")
    if not isinstance(bound_artifacts, Mapping):
        raise ValueError("bound_artifacts must map feature source, pipeline, and model names to files")
    expected_artifacts = {"feature_source", "feature_pipeline", *model_names}
    if set(bound_artifacts) != expected_artifacts:
        raise ValueError("bound_artifacts must provide the feature source, pipeline, and every model file")
    verified_paths: dict[str, str] = {}
    expected_hashes = {
        "feature_source": manifest["feature_source_sha256"],
        "feature_pipeline": manifest["feature_pipeline_sha256"],
        **model_hashes,
    }
    for name, expected_hash in expected_hashes.items():
        path = Path(bound_artifacts[name])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"signed calibration dependency is missing or unsafe: {name}")
        if _sha256(path) != expected_hash:
            raise ValueError(f"signed calibration dependency hash mismatch: {name}")
        verified_paths[name] = str(path)

    try:
        with np.load(artifact_path, allow_pickle=False) as stored:
            required = {"y", "entity_ids", "fold_ids"}
            prediction_keys = {f"prediction__{name}" for name in model_names}
            if not required.issubset(stored.files) or not prediction_keys.issubset(stored.files):
                raise ValueError("NPZ is missing target, entity, fold, or prediction arrays")
            allowed = required | prediction_keys | {"strata"}
            if set(stored.files) - allowed:
                raise ValueError("NPZ contains unrecognized arrays outside the signed schema")
            y = np.asarray(stored["y"], dtype=float).reshape(-1).copy()
            entity = np.asarray(stored["entity_ids"]).reshape(-1).copy()
            folds = np.asarray(stored["fold_ids"]).reshape(-1).copy()
            strata = (np.asarray(stored["strata"]).reshape(-1).copy()
                      if "strata" in stored.files else None)
            predictions = {
                name: np.asarray(stored[f"prediction__{name}"], dtype=float).reshape(-1).copy()
                for name in model_names
            }
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot safely load calibration NPZ: {exc}") from exc

    n = len(y)
    arrays = [entity, folds, *(predictions.values())]
    if strata is not None:
        arrays.append(strata)
    if n == 0 or any(len(array) != n for array in arrays):
        raise ValueError("calibration arrays must be nonempty and row-aligned")
    if not np.isfinite(y).all() or any(not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("target and predictions must be finite")
    if pd.isna(entity).any() or pd.Series(entity.astype(str)).str.strip().eq("").any():
        raise ValueError("entity IDs must be present and nonempty")
    if pd.isna(folds).any() or len(np.unique(folds)) < 2:
        raise ValueError("at least two complete OOF folds are required")
    if _array_sha256(folds) != manifest["fold_assignment_sha256"]:
        raise ValueError("fold assignments do not match the signed fold_assignment_sha256")
    if _array_sha256(entity.astype(str)) != manifest["entity_assignment_sha256"]:
        raise ValueError("entity assignments do not match the signed entity_assignment_sha256")
    if strata is not None and pd.isna(strata).any():
        raise ValueError("strata must be complete when supplied")
    entity_fold_counts = pd.DataFrame({"entity": entity.astype(str), "fold": folds}).groupby(
        "entity", sort=False, observed=True
    ).fold.nunique()
    if (entity_fold_counts > 1).any():
        raise ValueError("an entity appears in more than one OOF evaluation fold")

    # Prevent mutation through the returned object after signature verification.
    for array in [y, entity, folds, *(predictions.values())] + ([] if strata is None else [strata]):
        array.setflags(write=False)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return VerifiedCalibrationEvidence(
        y=y, predictions=predictions, entity_ids=entity, fold_ids=folds,
        strata=strata, model_sha256=dict(model_hashes),
        artifact_sha256=manifest["artifact_sha256"],
        manifest_sha256=manifest_sha256, reviewer_key_sha256=key_hash,
        split_protocol=split_protocol, prediction_time_rule=prediction_time_rule,
        artifact_path=str(artifact_path), manifest_path=str(manifest_path),
        reviewer_public_key_path=str(key_path),
        bound_artifacts=verified_paths,
    )


def fit_verified_case_risk_map(
    evidence: VerifiedCalibrationEvidence, *, pinned_reviewer_key_sha256: str, **kwargs,
) -> FrozenRiskMap:
    """Re-verify the signature against the deployment-pinned key, then fit.

    The caller must obtain ``pinned_reviewer_key_sha256`` from trusted
    deployment configuration, not from the evidence object or manifest.
    Re-loading here means a manually constructed/falsified dataclass cannot
    substitute arbitrary arrays for a valid signed artifact.
    """
    evidence = load_verified_calibration_evidence(
        evidence.artifact_path, evidence.manifest_path,
        evidence.reviewer_public_key_path, pinned_reviewer_key_sha256,
        bound_artifacts=evidence.bound_artifacts,
    )
    result = fit_case_risk_map(
        evidence.y, evidence.predictions, strata=evidence.strata,
        entity_ids=evidence.entity_ids,
        provenance={
            "source": "signature_verified_calibration_artifact",
            "artifact_sha256": evidence.artifact_sha256,
            "manifest_sha256": evidence.manifest_sha256,
            "reviewer_key_sha256": evidence.reviewer_key_sha256,
            "split_protocol": evidence.split_protocol,
            "prediction_time_rule": evidence.prediction_time_rule,
            **{f"model_sha256:{name}": digest for name, digest in evidence.model_sha256.items()},
        },
        **kwargs,
    )
    return replace(result, calibration_evidence_status="signature_verified_reviewer_attestation",
                   calibration_manifest_sha256=evidence.manifest_sha256,
                   model_sha256=dict(evidence.model_sha256))
