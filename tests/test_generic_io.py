import json

import numpy as np
import yaml

from pythia.cli import main
from pythia.io import load_certificate_config


def _write_pair_request(tmp_path, *, audited=True, concentration=None):
    rng = np.random.default_rng(52)
    n = 1000
    x = rng.uniform(size=n)
    d = rng.uniform(0, .01, n)
    artifact = tmp_path / "pairs.npz"
    np.savez(
        artifact, y_left=2 * x + rng.uniform(-.5, .5, n),
        y_right=2 * (x + d) + rng.uniform(-.5, .5, n), distances=d,
        left_entities=np.arange(n), right_entities=np.arange(n, 2 * n),
        pair_blocks=np.arange(n),
    )
    config = {
        "certificate": {
            "task": {"target": "y", "loss": "mae", "operational_unit": "minutes"},
            "representation": {
                "name": "fixed metric", "version": "1",
                "source_hash": "sha256:synthetic-metric",
                "provenance": {
                    "available_at_prediction_time": True,
                    "outcome_blind": True,
                },
            },
            "input": {
                "kind": "pair_matrix", "artifact": "pairs.npz",
                "meta": {
                    "pairing_outcome_blind": audited,
                    "iid_randomized_blocks": audited,
                    "conditionally_independent_entities": audited,
                },
            },
            "guard": {
                "kind": "replicate", "version": "rep-v1", "lipschitz_bound": 2,
                "distance_bound": .01, "outcome_difference_bound": 1.02,
                "calibration_population": "frozen synthetic contract",
                "source_hash": "sha256:guard",
            },
        }
    }
    if concentration is not None:
        config["certificate"]["guard"]["concentration"] = concentration
    path = tmp_path / "certificate.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_generic_pair_config_round_trip_and_cli(tmp_path, capsys):
    path = _write_pair_request(tmp_path)
    task, matrix, representation, guard = load_certificate_config(path)
    assert task.loss == "mae"
    assert matrix.y_left.size == 1000
    assert representation.source_hash
    assert guard.version == "rep-v1"
    assert main(["certify", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert output["claim_tier"] == "conditional"


def test_generic_pair_cli_refuses_unaudited_design(tmp_path, capsys):
    path = _write_pair_request(tmp_path, audited=False)
    assert main(["certify", str(path)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "refused"


def test_generic_pair_cli_refuses_missing_metric_source_hash(tmp_path, capsys):
    path = _write_pair_request(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload["certificate"]["representation"].pop("source_hash")
    path.write_text(yaml.safe_dump(payload))
    assert main(["certify", str(path)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert "source hash" in output["refusal"]["reason"]


def test_generic_pair_cli_auto_selects_conditional_rung(tmp_path, capsys):
    path = _write_pair_request(tmp_path, concentration="auto")
    assert main(["certify", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["diagnostics"]["automatic_concentration_selection"] is True
    assert output["diagnostics"]["selected_concentration"] == "conditional_empirical_bernstein"


def test_generic_loader_detects_artifact_hash_drift(tmp_path):
    path = _write_pair_request(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload["certificate"]["input"]["sha256"] = "wrong"
    path.write_text(yaml.safe_dump(payload))
    try:
        load_certificate_config(path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("artifact drift accepted")
