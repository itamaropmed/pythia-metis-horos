"""Canonical, auditable Mayo cohort and fold-manifest construction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class FoldManifestReport:
    accepted: bool
    algorithm: str
    source_hash: str
    manifest_hash: str
    n_eligible: int
    n_development: int
    n_holdout: int
    n_excluded_patient_history: int
    n_development_patients: int
    n_holdout_patients: int
    n_folds: int
    fold_sizes: tuple[int, ...]
    fold_patient_counts: tuple[int, ...]
    temporal_cutoff: str
    patient_overlap_count: int
    multi_fold_patient_count: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def build_grouped_temporal_fold_manifest(
    source_path: str | Path,
    output_path: str | Path,
    *,
    case_id_column: str = "case_durable_id",
    patient_column: str = "patient_durable_id",
    time_column: str = "scheduled_in_room",
    target_column: str = "case_actual_duration_time",
    test_fraction: float = 0.20,
    n_folds: int = 5,
) -> FoldManifestReport:
    """Build an outer temporal split and inner entity-grouped folds.

    The inner folds are patient-grouped but not forward-chaining temporal
    folds. This is intentional: M2 first reproduces the incumbent's exact
    historical evaluation contract. A changed temporal-CV design must be a
    separately versioned study.
    """
    if not 0.05 <= test_fraction <= 0.45:
        raise ValueError("test_fraction must be between 0.05 and 0.45")
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    source_path, output_path = Path(source_path), Path(output_path)
    required = [case_id_column, patient_column, time_column, target_column]
    frame = _read(source_path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"source lacks required columns: {missing}")
    if frame[case_id_column].isna().any() or frame[case_id_column].duplicated().any():
        raise ValueError("case IDs must be complete and unique")

    work = frame[required].copy()
    work["_source_row"] = np.arange(len(work), dtype=np.int64)
    work[time_column] = pd.to_datetime(work[time_column], errors="coerce", utc=True)
    target = pd.to_numeric(work[target_column], errors="coerce")
    eligible = work[time_column].notna() & target.notna() & target.gt(0) & work[patient_column].notna()
    if not eligible.all():
        raise ValueError(f"{int((~eligible).sum())} rows fail the frozen eligibility rule")
    work = work.sort_values([time_column, "_source_row"], kind="stable").reset_index(drop=True)
    cutoff = work[time_column].quantile(1.0 - test_fraction)
    late = work[time_column].ge(cutoff)
    holdout_patients = set(work.loc[late, patient_column].astype(str))
    patient_keys = work[patient_column].astype(str)
    is_holdout_patient = patient_keys.isin(holdout_patients)
    holdout = late & is_holdout_patient
    excluded_history = ~late & is_holdout_patient
    development = ~late & ~is_holdout_patient

    manifest = work[[case_id_column, patient_column, time_column, "_source_row"]].copy()
    manifest["outer_role"] = np.select(
        [development, holdout, excluded_history],
        ["development", "holdout", "excluded_patient_history"],
        default="ineligible",
    )
    manifest["fold_id"] = np.full(len(manifest), -1, dtype=np.int16)
    dev_idx = np.flatnonzero(development.to_numpy())
    groups = patient_keys.iloc[dev_idx].to_numpy()
    splitter = GroupKFold(n_splits=n_folds)
    for fold_id, (_, validation_local) in enumerate(
        splitter.split(np.zeros((len(dev_idx), 1)), groups=groups)
    ):
        manifest.loc[dev_idx[validation_local], "fold_id"] = fold_id
    # Holdout patients, including excluded earlier history, share one reserved
    # fold ID. Excluded history never enters fitting or evaluation.
    manifest.loc[is_holdout_patient, "fold_id"] = n_folds
    manifest["algorithm_version"] = "grouped-temporal-holdout+groupkfold-v1"
    manifest["temporal_cutoff"] = cutoff
    manifest["development_order"] = -1
    manifest.loc[dev_idx, "development_order"] = np.arange(len(dev_idx), dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)
    return audit_grouped_temporal_fold_manifest(
        output_path, source_hash=_hash(source_path), expected_folds=n_folds,
        case_id_column=case_id_column, patient_column=patient_column,
        time_column=time_column,
    )


def audit_grouped_temporal_fold_manifest(
    manifest_path: str | Path,
    *,
    source_hash: str = "unknown",
    expected_folds: int = 5,
    case_id_column: str = "case_durable_id",
    patient_column: str = "patient_durable_id",
    time_column: str = "scheduled_in_room",
) -> FoldManifestReport:
    manifest_path = Path(manifest_path)
    frame = _read(manifest_path)
    reasons: list[str] = []
    if frame[case_id_column].isna().any() or frame[case_id_column].duplicated().any():
        reasons.append("case IDs are null or non-unique")
    if frame["fold_id"].isna().any() or frame["fold_id"].lt(0).any():
        reasons.append("fold assignments are incomplete")
    dev = frame[frame["outer_role"].eq("development")]
    holdout = frame[frame["outer_role"].eq("holdout")]
    excluded = frame[frame["outer_role"].eq("excluded_patient_history")]
    dev_patients = set(dev[patient_column].astype(str))
    holdout_patients = set(holdout[patient_column].astype(str))
    overlap = len(dev_patients & holdout_patients)
    if overlap:
        reasons.append(f"{overlap} patients overlap development and holdout")
    patient_fold_counts = frame.groupby(patient_column)["fold_id"].nunique()
    multi_fold = int(patient_fold_counts.gt(1).sum())
    if multi_fold:
        reasons.append(f"{multi_fold} patients occur in multiple folds")
    cutoff_values = pd.to_datetime(frame["temporal_cutoff"], errors="coerce", utc=True).dropna().unique()
    if len(cutoff_values) != 1:
        reasons.append("manifest does not contain one valid temporal cutoff")
        cutoff = pd.NaT
    else:
        cutoff = pd.Timestamp(cutoff_values[0])
    times = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    if pd.notna(cutoff):
        if not times.loc[dev.index].lt(cutoff).all():
            reasons.append("development contains cases at or after the cutoff")
        if not times.loc[holdout.index].ge(cutoff).all():
            reasons.append("holdout contains cases before the cutoff")
    expected = set(range(expected_folds))
    observed = set(dev["fold_id"].astype(int).unique())
    if observed != expected:
        reasons.append(f"development folds are {sorted(observed)}, expected {sorted(expected)}")
    fold_sizes = tuple(int(dev["fold_id"].eq(fold).sum()) for fold in range(expected_folds))
    fold_patients = tuple(
        int(dev.loc[dev["fold_id"].eq(fold), patient_column].nunique())
        for fold in range(expected_folds)
    )
    return FoldManifestReport(
        accepted=not reasons,
        algorithm="grouped-temporal-holdout+groupkfold-v1",
        source_hash=source_hash,
        manifest_hash=_hash(manifest_path),
        n_eligible=int(len(frame)),
        n_development=int(len(dev)),
        n_holdout=int(len(holdout)),
        n_excluded_patient_history=int(len(excluded)),
        n_development_patients=len(dev_patients),
        n_holdout_patients=len(holdout_patients),
        n_folds=expected_folds,
        fold_sizes=fold_sizes,
        fold_patient_counts=fold_patients,
        temporal_cutoff=str(cutoff),
        patient_overlap_count=overlap,
        multi_fold_patient_count=multi_fold,
        reasons=tuple(reasons),
    )


def build_mayo_fold_manifest(
    source_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> FoldManifestReport:
    """Compatibility adapter for the Mayo column convention."""
    return build_grouped_temporal_fold_manifest(source_path, output_path, **kwargs)


def audit_mayo_fold_manifest(
    manifest_path: str | Path,
    **kwargs,
) -> FoldManifestReport:
    """Compatibility adapter for the Mayo column convention."""
    return audit_grouped_temporal_fold_manifest(manifest_path, **kwargs)
