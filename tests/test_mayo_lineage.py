import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from horos.models.mayo_lineage import (
    SPLIT_PROTOCOL, load_verified_mayo_inputs, ordered_case_id_sha256, read_verified_frames,
    verify_bundle, verify_loaded_frames, verify_split_cutoff,
)


def signed_bundle(tmp_path, *, target_dependency=False):
    if shutil.which("openssl") is None:
        pytest.skip("OpenSSL is required for signed lineage tests")
    directory = tmp_path / "derived"
    directory.mkdir()
    n = 12
    scheduled = pd.date_range("2020-01-01", periods=n, tz="UTC")
    frames = {
        "df.parquet": pd.DataFrame({
            "case_durable_id": [f"case-{i}" for i in range(n)],
            "patient_durable_id": [f"patient-{i}" for i in range(n)],
            "scheduled_in_room": scheduled,
            "feature_cutoff": scheduled - pd.Timedelta(days=2),
            "feature_available_at": scheduled - pd.Timedelta(days=4),
            "case_actual_duration_time": np.linspace(20., 90., n),
        }),
        "XA.parquet": pd.DataFrame({
            "case_durable_id": [f"case-{i}" for i in range(n)],
            "age": np.arange(n, dtype=float) ** 2 + 35,
        }),
        "XB.parquet": pd.DataFrame({
            "case_durable_id": [f"case-{i}" for i in range(n)],
            "age": np.arange(n, dtype=float) ** 2 + 35,
            "llm_concept": (np.sin(np.arange(n, dtype=float)) + 1.) / 2.,
        }),
    }
    hashes = {}
    for name, frame in frames.items():
        frame.to_parquet(directory / name, index=False)
        hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()

    private, public = tmp_path / "private.pem", tmp_path / "reviewer.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
                   check=True, capture_output=True)
    subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                   check=True, capture_output=True)
    source, generator, lock = (tmp_path / "source.bin", tmp_path / "features.py", tmp_path / "lock.txt")
    source.write_bytes(b"synthetic-source")
    generator.write_text("def build(frame): return frame[['age']]\n")
    lock.write_text("numpy==2.0\n")
    names = {"XA.parquet": ["age"], "XB.parquet": ["age", "llm_concept"]}
    manifest = {
        "artifact_sha256": hashes,
        "source_snapshot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_generation_code_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "environment_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "split_protocol": SPLIT_PROTOCOL, "lineage_version": 1,
        "feature_row_id_column": "case_durable_id",
        "split_cutoff_utc": "2020-01-10 00:00:00+00:00",
        "target_column": "case_actual_duration_time",
        "entity_column": "patient_durable_id", "time_column": "scheduled_in_room",
        "row_order_sha256": ordered_case_id_sha256(frames["df.parquet"].case_durable_id),
        "feature_columns": names,
        "features": {
            "age": {"source_fields": ["patient_age"], "transformation": "identity",
                    "target_dependency": target_dependency,
                    "available_at_prediction_time": True,
                    "availability_evidence": "source timestamp audit",
                    "available_at_column": "feature_available_at",
                    "prediction_cutoff_column": "feature_cutoff",
                    "prediction_cutoff_rule": "scheduled_in_room_minus_2_days"},
            "llm_concept": {"source_fields": ["preoperative_note"],
                            "transformation": "frozen reader v1",
                            "target_dependency": False,
                            "available_at_prediction_time": True,
                            "availability_evidence": "note cutoff audit",
                            "available_at_column": "feature_available_at",
                            "prediction_cutoff_column": "feature_cutoff",
                            "prediction_cutoff_rule": "scheduled_in_room_minus_2_days"},
        },
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    unsigned, signature_path = tmp_path / "unsigned.json", tmp_path / "signature.bin"
    unsigned.write_bytes(payload)
    subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin",
                    "-in", str(unsigned), "-out", str(signature_path)],
                   check=True, capture_output=True)
    manifest["signature_ed25519"] = base64.b64encode(signature_path.read_bytes()).decode()
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "directory": directory, "manifest": manifest_path, "key": public,
        "key_hash": hashlib.sha256(public.read_bytes()).hexdigest(),
        "source": source, "generator": generator, "lock": lock,
        "frames": frames,
    }


def verify(args):
    return verify_bundle(
        args["directory"], args["manifest"], args["key"], args["key_hash"],
        source_snapshot_path=args["source"], feature_code_path=args["generator"],
        environment_lock_path=args["lock"],
    )


def test_signed_bundle_is_loaded_and_verified(tmp_path):
    args = signed_bundle(tmp_path)
    bundle = verify(args)
    frame, xa, xb = read_verified_frames(bundle)
    verify_loaded_frames(bundle, frame, xa, xb)
    assert "case_durable_id" not in load_verified_mayo_inputs(SimpleNamespace(
        derived=args["directory"], lineage_manifest=args["manifest"],
        lineage_public_key=args["key"], lineage_public_key_sha256=args["key_hash"],
        source_snapshot=args["source"], feature_generation_code=args["generator"],
        environment_lock=args["lock"],
    ))[2].columns
    verify_split_cutoff(bundle, "2020-01-10 00:00:00+00:00")
    with pytest.raises(ValueError, match="cutoff differs"):
        verify_split_cutoff(bundle, "2020-01-11 00:00:00+00:00")


def test_tampered_derived_feature_file_refuses_before_read(tmp_path, monkeypatch):
    args = signed_bundle(tmp_path)
    (args["directory"] / "XB.parquet").write_bytes(b"replaced")
    namespace = SimpleNamespace(
        derived=args["directory"], lineage_manifest=args["manifest"],
        lineage_public_key=args["key"], lineage_public_key_sha256=args["key_hash"],
        source_snapshot=args["source"], feature_generation_code=args["generator"],
        environment_lock=args["lock"],
    )
    monkeypatch.setattr(
        pd, "read_parquet",
        lambda *a, **k: pytest.fail("parquet parsed before bundle verification"),
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_verified_mayo_inputs(namespace)


def test_target_dependent_signed_contract_is_refused(tmp_path):
    with pytest.raises(ValueError, match="target independence"):
        verify(signed_bundle(tmp_path, target_dependency=True))


def test_schema_and_row_order_must_match_manifest(tmp_path):
    args = signed_bundle(tmp_path)
    bundle = verify(args)
    frame, xa, xb = read_verified_frames(bundle)
    with pytest.raises(ValueError, match="schema/order"):
        verify_loaded_frames(bundle, frame, xa.rename(columns={"age": "renamed"}), xb)
    frame = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="row order"):
        verify_loaded_frames(bundle, frame, xa, xb)


def test_blank_patient_entity_ids_are_refused(tmp_path):
    args = signed_bundle(tmp_path)
    bundle = verify(args)
    frame, xa, xb = read_verified_frames(bundle)
    frame["patient_durable_id"] = "   "
    with pytest.raises(ValueError, match="patient entity IDs must be complete and nonempty"):
        verify_loaded_frames(bundle, frame, xa, xb)


def test_shuffled_feature_rows_are_refused_even_with_equal_shape(tmp_path):
    args = signed_bundle(tmp_path)
    bundle = verify(args)
    frame, xa, xb = read_verified_frames(bundle)
    shuffled = xa.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="feature row IDs"):
        verify_loaded_frames(bundle, frame, shuffled, xb)


def test_exact_and_affine_target_copy_are_refused(tmp_path):
    args = signed_bundle(tmp_path)
    bundle = verify(args)
    frame, xa, xb = read_verified_frames(bundle)
    y = frame.case_actual_duration_time.to_numpy()
    with pytest.raises(ValueError, match="exactly equals the target"):
        verify_loaded_frames(bundle, frame, xa.assign(age=y), xb.assign(age=y))
    with pytest.raises(ValueError, match="affine target copy"):
        verify_loaded_frames(bundle, frame, xa, xb.assign(llm_concept=2.5 * y + 7))


def test_noisy_target_proxy_can_evade_value_only_tripwire(tmp_path):
    """Signed semantic assertions, not numeric checks, must catch noisy proxies."""
    args = signed_bundle(tmp_path)
    bundle = verify(args)
    frame, xa, xb = read_verified_frames(bundle)
    y = frame.case_actual_duration_time.to_numpy()
    jitter = np.array([0., .2, -.1, .4, -.3, .1, -.2, .3, .05, -.4, .25, -.05])
    poisoned = xb.assign(llm_concept=y + jitter)
    # The verifier can catch exact/affine copies, but cannot infer that this
    # deliberately false feature's signed semantic claim is untrue.
    verify_loaded_frames(bundle, frame, xa, poisoned)


def test_feature_timestamp_after_case_cutoff_is_refused(tmp_path):
    args = signed_bundle(tmp_path)
    bundle = verify(args)
    frame, xa, xb = read_verified_frames(bundle)
    future_frame = frame.copy()
    future_frame["feature_available_at"] = future_frame["feature_cutoff"] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="not available by the declared prediction cutoff"):
        verify_loaded_frames(bundle, future_frame, xa, xb)


def test_signed_feature_cannot_choose_a_later_cutoff(tmp_path):
    args = signed_bundle(tmp_path)
    bundle = verify(args)
    frame, xa, xb = read_verified_frames(bundle)
    late_cutoff = frame.copy()
    late_cutoff["feature_cutoff"] = late_cutoff["scheduled_in_room"] + pd.Timedelta(days=10)
    late_cutoff["feature_available_at"] = late_cutoff["feature_cutoff"] - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="cutoff does not match its scheduled-time rule"):
        verify_loaded_frames(bundle, late_cutoff, xa, xb)


@pytest.mark.parametrize("script", [
    "run_mayo_metis_case_risk.py", "run_mayo_metis_case_risk_v2.py",
    "run_mayo_pythia_local_w30.py", "run_mayo_pythia_local_w30_v2.py",
])
def test_primary_mayo_entrypoint_requires_preparse_lineage_gate(script):
    source = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()
    assert "add_lineage_arguments(" in source
    assert "load_verified_mayo_inputs(args)" in source
    assert "verify_split_cutoff(lineage, cutoff)" in source
    assert "pd.read_parquet(" not in source
