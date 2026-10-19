import copy

import pytest

from pythia import canonical_manifest_hash, compile_certificate_manifest


def frozen_manifest():
    return {
        "schema_version": 1,
        "status": "frozen",
        "context": {
            "target": "duration", "population": "synthetic audit",
            "loss": "MAE", "outcome_scale": "minutes",
            "decision_time": "pre-outcome", "outcome_clip": [0., 300.],
        },
        "target_node": "raw-notes",
        "family_alpha": .05,
        "permitted_tiers": ["proved", "conditional"],
        "lower_evidence": [{
            "evidence_id": "base", "node": "tabular", "lower": 30.,
            "claim_tier": "conditional", "source": "SAA",
            "provenance": {"hash": "abc"}, "event_id": "base-family",
            "failure_probability": .02, "assumptions": ["stable groups"],
        }],
        "gap_evidence": [{
            "evidence_id": "gap", "poorer": "tabular", "richer": "raw-notes",
            "upper_gap": 4., "claim_tier": "conditional",
            "kind": "declared_sensitivity", "source": "external declaration",
            "provenance": {"version": "v1"}, "failure_probability": 0.,
            "assumptions": ["declared cap is valid"],
        }],
        "metadata": {"purpose": "test"},
        "unresolved": [],
    }


def test_frozen_manifest_compiles_and_is_content_addressed():
    manifest = frozen_manifest()
    result = compile_certificate_manifest(manifest)
    assert result["certificate"]["status"] == "complete"
    assert result["certificate"]["bound"] == 26.
    assert result["certificate"]["diagnostics"]["manifest_hash"] == result["manifest_hash"]
    reordered = {key: manifest[key] for key in reversed(tuple(manifest))}
    assert canonical_manifest_hash(reordered) == result["manifest_hash"]
    changed = copy.deepcopy(manifest)
    changed["gap_evidence"][0]["upper_gap"] = 5.
    assert canonical_manifest_hash(changed) != result["manifest_hash"]


def test_draft_and_unresolved_frozen_manifest_refuse():
    draft = frozen_manifest()
    draft["status"] = "draft"
    draft["unresolved"] = ["raw-note gap"]
    result = compile_certificate_manifest(draft)
    assert result["certificate"]["status"] == "refused"
    assert result["certificate"]["diagnostics"]["unresolved"] == ("raw-note gap",)
    frozen = frozen_manifest()
    frozen["unresolved"] = ["cohort hash"]
    assert compile_certificate_manifest(frozen)["certificate"]["status"] == "refused"


def test_manifest_rejects_unknown_fields_and_nonfinite_json():
    unknown = frozen_manifest()
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="unknown manifest"):
        compile_certificate_manifest(unknown)
    nonfinite = frozen_manifest()
    nonfinite["family_alpha"] = float("nan")
    with pytest.raises(ValueError, match="finite canonical JSON"):
        compile_certificate_manifest(nonfinite)
    nested = frozen_manifest()
    nested["lower_evidence"][0]["unreviewed_override"] = True
    with pytest.raises(ValueError, match="invalid lower evidence fields"):
        compile_certificate_manifest(nested)
