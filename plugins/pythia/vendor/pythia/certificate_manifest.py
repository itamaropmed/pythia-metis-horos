"""Strict, content-hashed manifests for certificate-graph compilation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .certificate_graph import (
    CompiledCertificate,
    GapEvidence,
    LowerEvidence,
    compile_certificate_graph,
)
from .evidence_adapters import EvidenceContext


_TOP_LEVEL = {
    "schema_version", "status", "context", "target_node", "family_alpha",
    "permitted_tiers", "lower_evidence", "gap_evidence", "metadata", "unresolved",
}
_LOWER_REQUIRED = {"evidence_id", "node", "lower", "claim_tier", "source", "provenance"}
_LOWER_OPTIONAL = {"event_id", "failure_probability", "assumptions"}
_GAP_REQUIRED = {
    "evidence_id", "poorer", "richer", "upper_gap", "claim_tier",
    "kind", "source", "provenance",
}
_GAP_OPTIONAL = {"event_id", "failure_probability", "assumptions"}


def canonical_manifest_hash(manifest: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest must be finite canonical JSON") from exc
    return hashlib.sha256(payload.encode()).hexdigest()


def _refusal(target: str, reason: str, referral: str, manifest_hash: str,
             diagnostics: Mapping[str, object] | None = None) -> dict:
    certificate = CompiledCertificate(
        status="refused", claim_tier="refused", target_node=target,
        bound=None, family_failure_probability=0.,
        diagnostics=dict(diagnostics or {}),
        refusal={"reason": reason, "referral": referral})
    return {"manifest_hash": manifest_hash, "certificate": certificate.to_dict()}


def _validate_record(values: Mapping[str, object], required: set[str],
                     optional: set[str], label: str) -> None:
    missing = required - set(values)
    unknown = set(values) - required - optional
    if missing or unknown:
        raise ValueError(
            f"invalid {label} fields; missing={sorted(missing)}, unknown={sorted(unknown)}")


def compile_certificate_manifest(manifest: Mapping[str, object]) -> dict:
    """Compile one frozen aggregate-only manifest into an auditable result."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    unknown = set(manifest) - _TOP_LEVEL
    if unknown:
        raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
    digest = canonical_manifest_hash(manifest)
    target = str(manifest.get("target_node", ""))
    if manifest.get("schema_version") != 1:
        return _refusal(target, "unsupported certificate manifest schema",
                        "migrate and freeze a schema-version-1 manifest", digest)
    if manifest.get("status") != "frozen":
        unresolved = tuple(map(str, manifest.get("unresolved", ())))
        return _refusal(
            target, "certificate manifest is not frozen",
            "resolve blockers, freeze all evidence choices, and recompute the hash", digest,
            {"manifest_status": manifest.get("status"), "unresolved": unresolved})
    if manifest.get("unresolved"):
        return _refusal(
            target, "frozen manifest still declares unresolved evidence",
            "resolve every required edge or retain draft status", digest,
            {"unresolved": tuple(map(str, manifest["unresolved"]))})
    try:
        context = EvidenceContext(**dict(manifest["context"]))
        problem_id = context.problem_id
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest context is invalid") from exc
    lowers = []
    for row in manifest.get("lower_evidence", ()):
        values = dict(row)
        _validate_record(values, _LOWER_REQUIRED, _LOWER_OPTIONAL, "lower evidence")
        lowers.append(LowerEvidence(
            evidence_id=str(values["evidence_id"]), problem_id=problem_id,
            node=str(values["node"]), lower=float(values["lower"]),
            claim_tier=str(values["claim_tier"]), source=str(values["source"]),
            provenance=dict(values["provenance"]),
            event_id=None if values.get("event_id") is None else str(values["event_id"]),
            failure_probability=float(values.get("failure_probability", 0.)),
            assumptions=tuple(map(str, values.get("assumptions", ()))),
        ))
    gaps = []
    for row in manifest.get("gap_evidence", ()):
        values = dict(row)
        _validate_record(values, _GAP_REQUIRED, _GAP_OPTIONAL, "gap evidence")
        gaps.append(GapEvidence(
            evidence_id=str(values["evidence_id"]), problem_id=problem_id,
            poorer=str(values["poorer"]), richer=str(values["richer"]),
            upper_gap=float(values["upper_gap"]),
            claim_tier=str(values["claim_tier"]), kind=str(values["kind"]),
            source=str(values["source"]), provenance=dict(values["provenance"]),
            event_id=None if values.get("event_id") is None else str(values["event_id"]),
            failure_probability=float(values.get("failure_probability", 0.)),
            assumptions=tuple(map(str, values.get("assumptions", ()))),
        ))
    certificate = compile_certificate_graph(
        target_node=target, lower_evidence=tuple(lowers), gap_evidence=tuple(gaps),
        family_alpha=float(manifest.get("family_alpha", .05)),
        permitted_tiers=tuple(map(str, manifest.get(
            "permitted_tiers", ("proved", "conditional")))))
    result = certificate.to_dict()
    result["diagnostics"] = {
        **result["diagnostics"], "manifest_hash": digest,
        "problem_id": problem_id, "schema_version": 1,
    }
    return {"manifest_hash": digest, "problem_id": problem_id,
            "certificate": result, "metadata": dict(manifest.get("metadata", {}))}


def compile_certificate_manifest_file(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return compile_certificate_manifest(manifest)
