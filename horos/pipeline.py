"""Config-driven preparation stage for reusable certification studies."""
from __future__ import annotations

from pathlib import Path

from pythia.folds import build_grouped_temporal_fold_manifest

from .config import load_study


def prepare_study(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    study = load_study(config_path)
    root = config_path.parent
    source = (root / study.dataset.path).resolve()
    manifest = (root / study.split.manifest).resolve()
    folds = build_grouped_temporal_fold_manifest(
        source, manifest,
        case_id_column=study.dataset.row_id,
        patient_column=study.dataset.group,
        time_column=study.dataset.time,
        target_column=study.dataset.target,
        test_fraction=study.split.test_fraction,
        n_folds=study.split.inner_folds,
    )
    return {
        "status": "prepared" if folds.accepted else "refused",
        "study": study.name,
        "loss": study.loss,
        "dataset_roles": {
            "row_id": study.dataset.row_id, "target": study.dataset.target,
            "group": study.dataset.group, "time": study.dataset.time,
        },
        "fold_manifest": folds.to_dict(),
    }
