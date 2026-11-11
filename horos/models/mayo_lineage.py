"""Fail-closed lineage verification for precomputed Mayo feature bundles.

Cryptographic integrity does not prove feature semantics. The signature key
must be pinned independently of the bundle, and the signer must attest to
target independence and prediction-time availability per feature.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_FILES = ("df.parquet", "XA.parquet", "XB.parquet")
SPLIT_PROTOCOL = "mayo_hrs_temporal_p80_patient_disjoint_v1"
BLOCKED_NAMES = {
    "case_durable_id", "patient_durable_id", "fold_id", "patient_birth_date",
    "case_actual_duration_time", "case_actual_patient_in_room",
    "case_actual_prep_time", "case_actual_procedure_time",
    "case_actual_wrapup_time", "all_procs_not_performed",
    "scheduled_setup_start", "scheduled_in_room", "scheduled_out_of_room",
    "scheduled_cleanup_complete",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blocked(name: object) -> bool:
    canonical = str(name).strip().casefold()
    blocked = {value.casefold() for value in BLOCKED_NAMES}
    return canonical in blocked or canonical.startswith("case_actual_") or canonical.endswith("_date")


def _availability_precedes_cutoff(frame: pd.DataFrame, contract: Mapping[str, object], name: str) -> bool:
    available_col = contract.get("available_at_column")
    cutoff_col = contract.get("prediction_cutoff_column")
    if (not isinstance(available_col, str) or not isinstance(cutoff_col, str)
            or available_col not in frame or cutoff_col not in frame):
        raise ValueError(f"feature {name!r} lacks row-level availability/cutoff columns")
    available_raw, cutoff_raw = frame[available_col], frame[cutoff_col]
    if (pd.api.types.is_bool_dtype(available_raw.dtype)
            or pd.api.types.is_bool_dtype(cutoff_raw.dtype)):
        raise ValueError(f"feature {name!r} has boolean availability timestamps")
    numeric = pd.api.types.is_numeric_dtype(available_raw.dtype)
    if numeric != pd.api.types.is_numeric_dtype(cutoff_raw.dtype):
        raise ValueError(f"feature {name!r} mixes availability timestamp types")
    rule = contract.get("prediction_cutoff_rule")
    scheduled = pd.to_datetime(frame["scheduled_in_room"], errors="coerce", utc=True)
    if scheduled.isna().any():
        raise ValueError(f"feature {name!r} cannot bind cutoff to scheduled_in_room")
    if rule == "scheduled_in_room":
        expected = scheduled
    elif rule == "scheduled_in_room_minus_2_days":
        expected = scheduled - pd.Timedelta(days=2)
    else:
        raise ValueError(f"feature {name!r} has unsupported prediction_cutoff_rule")
    if numeric:
        units = {"unix_seconds": 1e9, "unix_milliseconds": 1e6,
                 "unix_microseconds": 1e3, "unix_nanoseconds": 1.}
        unit = contract.get("time_unit")
        if unit not in units:
            raise ValueError(f"feature {name!r} numeric timestamps require a supported time_unit")
        available = pd.to_numeric(available_raw, errors="coerce").to_numpy(float)
        cutoff = pd.to_numeric(cutoff_raw, errors="coerce").to_numpy(float)
        expected_numeric = expected.astype("int64").to_numpy(float) / units[unit]
        if (not np.isfinite(available).all() or not np.isfinite(cutoff).all()
                or not np.allclose(cutoff, expected_numeric, rtol=0., atol=1e-6)):
            raise ValueError(f"feature {name!r} cutoff does not match its scheduled-time rule")
        return bool(np.all(available <= expected_numeric))
    available = pd.to_datetime(available_raw, errors="coerce", utc=True)
    cutoff = pd.to_datetime(cutoff_raw, errors="coerce", utc=True)
    if available.isna().any() or cutoff.isna().any():
        raise ValueError(f"feature {name!r} has missing/unparseable availability timestamps")
    if not cutoff.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise ValueError(f"feature {name!r} cutoff does not match its scheduled-time rule")
    return bool((available <= expected).all())


def verify_signed_manifest(manifest_path: Path, key_path: Path, pinned_key_sha256: str):
    key_hash = _hash(key_path)
    if key_hash != pinned_key_sha256:
        raise ValueError("lineage public key hash does not match the pinned trust anchor")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Mayo lineage manifest must be a JSON object")
    signature_text = manifest.get("signature_ed25519")
    if not isinstance(signature_text, str):
        raise ValueError("Mayo lineage manifest requires signature_ed25519")
    try:
        signature = base64.b64decode(signature_text, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Mayo lineage signature is not valid base64") from exc
    unsigned = {key: value for key, value in manifest.items() if key != "signature_ed25519"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("OpenSSL is required to verify Mayo lineage signatures")
    with tempfile.TemporaryDirectory(prefix="mayo-lineage-") as temp:
        payload_path, signature_path = Path(temp) / "manifest.json", Path(temp) / "signature.bin"
        payload_path.write_bytes(payload)
        signature_path.write_bytes(signature)
        result = subprocess.run(
            [openssl, "pkeyutl", "-verify", "-pubin", "-inkey", str(key_path),
             "-rawin", "-in", str(payload_path), "-sigfile", str(signature_path)],
            capture_output=True, check=False,
        )
    if result.returncode:
        raise ValueError("Mayo lineage manifest signature verification failed")
    return unsigned, key_hash


@dataclass(frozen=True)
class VerifiedBundle:
    manifest: Mapping[str, object]
    manifest_sha256: str
    reviewer_key_sha256: str
    directory: Path
    artifact_sha256: Mapping[str, str]


def verify_bundle(
    derived_dir: str | Path, manifest_path: str | Path, public_key_path: str | Path,
    pinned_key_sha256: str, *, source_snapshot_path: str | Path,
    feature_code_path: str | Path, environment_lock_path: str | Path,
) -> VerifiedBundle:
    """Verify all artifacts and signed semantic claims before parsing parquet."""
    directory, manifest_path, public_key_path = map(Path, (derived_dir, manifest_path, public_key_path))
    manifest, key_hash = verify_signed_manifest(manifest_path, public_key_path, pinned_key_sha256)
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(REQUIRED_FILES):
        raise ValueError("signed manifest must bind exactly df.parquet, XA.parquet, and XB.parquet")
    for name in REQUIRED_FILES:
        artifact = directory / name
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"required Mayo artifact is missing or not a regular file: {name}")
        if _hash(artifact) != hashes[name]:
            raise ValueError(f"signed Mayo artifact hash mismatch: {name}")
    for field, path_value in (
        ("source_snapshot_sha256", source_snapshot_path),
        ("feature_generation_code_sha256", feature_code_path),
        ("environment_lock_sha256", environment_lock_path),
    ):
        path = Path(path_value)
        if not path.is_file() or manifest.get(field) != _hash(path):
            raise ValueError(f"signed Mayo manifest {field} mismatch or missing artifact")
    if manifest.get("split_protocol") != SPLIT_PROTOCOL:
        raise ValueError("signed Mayo manifest has the wrong split protocol")
    for field, expected in (("target_column", "case_actual_duration_time"),
                            ("entity_column", "patient_durable_id"),
                            ("time_column", "scheduled_in_room")):
        if manifest.get(field) != expected:
            raise ValueError(f"signed Mayo manifest does not bind expected {field}")
    if manifest.get("lineage_version") != 1:
        raise ValueError("unsupported Mayo lineage manifest version")
    if manifest.get("feature_row_id_column") != "case_durable_id":
        raise ValueError("signed Mayo artifacts must carry case_durable_id in XA and XB")
    cutoff = manifest.get("split_cutoff_utc")
    if not isinstance(cutoff, str) or not cutoff.strip():
        raise ValueError("signed Mayo manifest must bind the realized UTC split cutoff")
    for field in ("row_order_sha256", "source_snapshot_sha256",
                  "feature_generation_code_sha256", "environment_lock_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"signed Mayo manifest requires valid SHA-256: {field}")
    columns = manifest.get("feature_columns")
    contracts = manifest.get("features")
    if not isinstance(columns, Mapping) or set(columns) != {"XA.parquet", "XB.parquet"}:
        raise ValueError("signed Mayo manifest must list ordered XA/XB columns")
    if not isinstance(contracts, Mapping):
        raise ValueError("signed Mayo manifest requires per-feature lineage contracts")
    names = []
    for filename in ("XA.parquet", "XB.parquet"):
        declared = columns[filename]
        if not isinstance(declared, list) or not declared or len(declared) != len(set(declared)):
            raise ValueError(f"invalid signed feature schema for {filename}")
        if any(not isinstance(name, str) or not name.strip() for name in declared):
            raise ValueError(f"invalid signed feature name for {filename}")
        names.extend(declared)
    if set(contracts) != set(names):
        # Shared tabular columns may appear in both arms; contracts are keyed by
        # feature name, so compare the de-duplicated union.
        if set(contracts) != set(names):
            raise ValueError("feature contracts do not match declared feature columns")
    for name, contract in contracts.items():
        if _blocked(name) or not isinstance(contract, Mapping):
            raise ValueError(f"invalid or reserved feature lineage contract: {name}")
        sources = contract.get("source_fields")
        if not isinstance(sources, list) or not sources or any(not str(v).strip() for v in sources):
            raise ValueError(f"feature contract lacks source fields: {name}")
        if not str(contract.get("transformation", "")).strip():
            raise ValueError(f"feature contract lacks transformation: {name}")
        if contract.get("target_dependency") is not False:
            raise ValueError(f"feature contract does not attest target independence: {name}")
        if contract.get("available_at_prediction_time") is not True:
            raise ValueError(f"feature contract does not attest as-of availability: {name}")
        if not str(contract.get("availability_evidence", "")).strip():
            raise ValueError(f"feature contract lacks availability evidence: {name}")
        if not isinstance(contract.get("available_at_column"), str):
            raise ValueError(f"feature contract lacks an availability timestamp column: {name}")
        if not isinstance(contract.get("prediction_cutoff_column"), str):
            raise ValueError(f"feature contract lacks a prediction cutoff column: {name}")
        if contract.get("prediction_cutoff_rule") not in {
            "scheduled_in_room", "scheduled_in_room_minus_2_days",
        }:
            raise ValueError(f"feature contract has unsupported prediction cutoff rule: {name}")
    return VerifiedBundle(manifest, _hash(manifest_path), key_hash, directory, hashes)


def _hash_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def read_verified_frames(bundle: VerifiedBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hash, parse, and re-hash each same file descriptor to prevent path swaps."""
    frames = []
    for name in REQUIRED_FILES:
        path = bundle.directory / name
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"signed Mayo artifact is not a regular file: {name}")
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if _hash_stream(stream) != bundle.artifact_sha256[name]:
                raise ValueError(f"signed Mayo artifact hash mismatch at load: {name}")
            stream.seek(0)
            frame = pd.read_parquet(stream)
            stream.seek(0)
            after_hash = _hash_stream(stream)
            after = os.fstat(stream.fileno())
            if after_hash != bundle.artifact_sha256[name] or identity != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise ValueError(f"signed Mayo artifact changed during load: {name}")
            frames.append(frame)
    return frames[0], frames[1], frames[2]


def ordered_case_id_sha256(case_ids) -> str:
    return hashlib.sha256(json.dumps(
        [str(value) for value in case_ids], separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def verify_loaded_frames(bundle: VerifiedBundle, frame: pd.DataFrame,
                         xa: pd.DataFrame, xb: pd.DataFrame) -> None:
    """Check signed column order, alignment, and exact/affine target aliases."""
    required_roles = {"case_durable_id", "patient_durable_id", "scheduled_in_room",
                      "case_actual_duration_time"}
    missing = sorted(required_roles - set(frame.columns))
    if missing:
        raise ValueError(f"Mayo case table lacks required lineage roles: {missing}")
    if frame["case_durable_id"].isna().any() or frame["case_durable_id"].duplicated().any():
        raise ValueError("case IDs must be complete and unique")
    patient_ids = frame["patient_durable_id"]
    if patient_ids.isna().any() or patient_ids.astype(str).str.strip().eq("").any():
        raise ValueError("patient entity IDs must be complete and nonempty")
    scheduled = pd.to_datetime(frame["scheduled_in_room"], errors="coerce", utc=True)
    if scheduled.isna().any():
        raise ValueError("scheduled timestamps must be complete and parseable")
    target_values = pd.to_numeric(frame["case_actual_duration_time"], errors="coerce")
    if target_values.isna().any() or not np.isfinite(target_values.to_numpy(float)).all():
        raise ValueError("duration targets must be complete and finite")
    declared = bundle.manifest["feature_columns"]
    row_id_column = bundle.manifest["feature_row_id_column"]
    if (list(xa.columns) != [row_id_column, *declared["XA.parquet"]]
            or list(xb.columns) != [row_id_column, *declared["XB.parquet"]]):
        raise ValueError("loaded XA/XB feature schema/order differs from signed manifest")
    if len({len(frame), len(xa), len(xb)}) != 1:
        raise ValueError("df/XA/XB row counts differ")
    case_ids = frame.case_durable_id.astype(str).tolist()
    if ordered_case_id_sha256(case_ids) != bundle.manifest["row_order_sha256"]:
        raise ValueError("case row order differs from signed feature attestation")
    if (xa[row_id_column].astype(str).tolist() != case_ids
            or xb[row_id_column].astype(str).tolist() != case_ids):
        raise ValueError("XA/XB feature row IDs do not match the ordered case table")
    y = target_values.to_numpy(float)
    for name, contract in bundle.manifest["features"].items():
        if not _availability_precedes_cutoff(frame, contract, name):
            raise ValueError(f"feature {name!r} was not available by the declared prediction cutoff")
    for filename, features in (("XA.parquet", xa), ("XB.parquet", xb)):
        for name in declared[filename]:
            if _blocked(name):
                raise ValueError(f"reserved or outcome-like feature name: {name}")
            x = pd.to_numeric(features[name], errors="coerce").to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 3 or np.ptp(y[valid]) <= 1e-12:
                continue
            if np.array_equal(x[valid], y[valid]):
                raise ValueError(f"feature {name!r} in {filename} exactly equals the target")
            if valid.sum() < 8 or np.ptp(x[valid]) <= 1e-12:
                continue
            slope, intercept = np.linalg.lstsq(np.column_stack([y[valid], np.ones(valid.sum())]), x[valid], rcond=None)[0]
            residual = x[valid] - (slope * y[valid] + intercept)
            if np.max(np.abs(residual)) <= 1e-12 * max(1., float(np.max(np.abs(x[valid])))):
                raise ValueError(f"feature {name!r} in {filename} is an affine target copy")


def verify_split_cutoff(bundle: VerifiedBundle, cutoff: str) -> None:
    if bundle.manifest["split_cutoff_utc"] != cutoff:
        raise ValueError("executed Mayo temporal cutoff differs from signed manifest")


def add_lineage_arguments(parser) -> None:
    """Require independent provenance inputs on every production Mayo runner."""
    parser.add_argument("--lineage-manifest", type=Path, required=True)
    parser.add_argument("--lineage-public-key", type=Path, required=True)
    parser.add_argument("--lineage-public-key-sha256", required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--feature-generation-code", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)


def load_verified_mayo_inputs(args):
    """Verify signer, hashes, and feature contract before parsing patient rows."""
    bundle = verify_bundle(
        args.derived, args.lineage_manifest, args.lineage_public_key,
        args.lineage_public_key_sha256,
        source_snapshot_path=args.source_snapshot,
        feature_code_path=args.feature_generation_code,
        environment_lock_path=args.environment_lock,
    )
    frame, xa, xb = read_verified_frames(bundle)
    verify_loaded_frames(bundle, frame, xa, xb)
    row_id_column = bundle.manifest["feature_row_id_column"]
    return bundle, frame, xa.drop(columns=[row_id_column]), xb.drop(columns=[row_id_column])
