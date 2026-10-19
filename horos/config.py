"""Declarative study configuration for dataset-agnostic Pythia workflows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    row_id: str
    target: str
    group: str
    time: str


@dataclass(frozen=True)
class SplitSpec:
    kind: str = "grouped_temporal_holdout"
    test_fraction: float = 0.20
    inner_folds: int = 5
    manifest: str = "outputs/fold_manifest.parquet"


@dataclass(frozen=True)
class StudySpec:
    name: str
    loss: str
    dataset: DatasetSpec
    split: SplitSpec


def load_study(path: str | Path) -> StudySpec:
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("study configuration must be a mapping")
    required = {"name", "loss", "dataset", "split"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"study configuration lacks: {missing}")
    dataset = DatasetSpec(**payload["dataset"])
    split = SplitSpec(**payload["split"])
    if split.kind != "grouped_temporal_holdout":
        raise ValueError(f"unsupported split kind: {split.kind!r}")
    if not all((dataset.path, dataset.row_id, dataset.target, dataset.group, dataset.time)):
        raise ValueError("dataset path and column roles must be non-empty")
    return StudySpec(
        name=str(payload["name"]), loss=str(payload["loss"]),
        dataset=dataset, split=split,
    )
