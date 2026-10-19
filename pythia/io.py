"""Deterministic, model-free loading of generic certificate configurations."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from .contracts import (
    DependenceGuardSpec, FoldMatrix, PairMatrix, RangeDependenceGuardSpec,
    ReplicateGuardSpec, Representation, TaskCard,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(archive, name: str) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"certificate artifact is missing required array '{name}'")
    return archive[name]


def load_certificate_config(path: str | Path):
    """Load a generic YAML + NPZ certificate request without training models."""
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or "certificate" not in payload:
        raise ValueError("generic certificate config requires a 'certificate' mapping")
    config = payload["certificate"]
    artifact_path = (config_path.parent / config["input"]["artifact"]).resolve()
    artifact_hash = _sha256(artifact_path)
    expected_hash = config["input"].get("sha256")
    if expected_hash and expected_hash != artifact_hash:
        raise ValueError("certificate input artifact hash mismatch")

    task = TaskCard(**config["task"])
    representation_values = dict(config["representation"])
    # A pair artifact hash is not the source hash of the representation that
    # created its distances. Require that separate provenance explicitly.
    if config["input"]["kind"] != "pair_matrix":
        representation_values.setdefault("source_hash", artifact_hash)
    if "feature_names" in representation_values:
        representation_values["feature_names"] = tuple(representation_values["feature_names"])
    representation = Representation(**representation_values)
    input_config = config["input"]
    meta = dict(input_config.get("meta", {}))
    meta["artifact_sha256"] = artifact_hash
    with np.load(artifact_path, allow_pickle=False) as archive:
        kind = input_config["kind"]
        if kind == "fold_matrix":
            p = _required(archive, "P_oof")
            names = tuple(str(value) for value in _required(archive, "names"))
            families = tuple(str(value) for value in _required(archive, "families"))
            matrix = FoldMatrix(
                y=_required(archive, "y"), P_oof=p, names=names, families=families,
                fold_id=_required(archive, "fold_id"),
                groups=archive["groups"] if "groups" in archive else None,
                row_ids=archive["row_ids"] if "row_ids" in archive else None,
                meta=meta,
            )
        elif kind == "pair_matrix":
            matrix = PairMatrix(
                y_left=_required(archive, "y_left"),
                y_right=_required(archive, "y_right"),
                distances=_required(archive, "distances"),
                left_entities=_required(archive, "left_entities"),
                right_entities=_required(archive, "right_entities"),
                pair_blocks=_required(archive, "pair_blocks"), meta=meta,
            )
        else:
            raise ValueError("input kind must be 'fold_matrix' or 'pair_matrix'")

    guard_values = dict(config["guard"])
    guard_kind = guard_values.pop("kind")
    guard_types = {
        "replicate": ReplicateGuardSpec,
        "dependence_finite_variance": DependenceGuardSpec,
        "dependence_bounded_range": RangeDependenceGuardSpec,
    }
    if guard_kind not in guard_types:
        raise ValueError(f"unsupported generic guard kind '{guard_kind}'")
    guard = guard_types[guard_kind](**guard_values)
    return task, matrix, representation, guard
