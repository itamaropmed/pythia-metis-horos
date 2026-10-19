"""Fail-closed audit for importing an incumbent prediction artifact."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IncumbentAudit:
    accepted: bool
    artifact_hash: str
    n_rows: int
    n_groups: int
    mae: float
    expected_mae: float
    mae_tolerance: float
    has_fold_provenance: bool
    manifest_hash: str | None
    cohort_matches_manifest: bool | None
    folds_match_manifest: bool | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_incumbent(path: str | Path, *, target_column: str,
                    prediction_column: str, row_id_column: str,
                    group_column: str, expected_mae: float = 31.2623,
                    fold_column: str = "fold_id", tolerance: float = 1e-4,
                    require_oof: bool = True,
                    manifest_path: str | Path | None = None,
                    manifest_role: str = "development") -> IncumbentAudit:
    path = Path(path)
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    required = {target_column, prediction_column, row_id_column, group_column}
    missing = sorted(required - set(frame.columns))
    reasons = []
    if missing:
        raise ValueError(f"incumbent artifact lacks required columns: {missing}")
    if frame[row_id_column].isna().any() or frame[row_id_column].duplicated().any():
        reasons.append("row IDs are null or non-unique")
    values = frame[[target_column, prediction_column]].to_numpy(float)
    if not np.isfinite(values).all():
        reasons.append("target or prediction contains non-finite values")
    mae = float(np.mean(np.abs(values[:, 0] - values[:, 1])))
    if abs(mae - expected_mae) > tolerance:
        reasons.append(
            f"MAE {mae:.10f} does not reproduce expected {expected_mae:.10f} "
            f"within tolerance {tolerance:g}"
        )
    has_fold = fold_column in frame and frame[fold_column].notna().all()
    if require_oof and not has_fold:
        reasons.append(f"missing complete {fold_column!r} provenance; held-out predictions are not OOF")
    manifest_hash = None
    cohort_matches: bool | None = None
    folds_match: bool | None = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest = pd.read_csv(manifest_path) if manifest_path.suffix.lower() == ".csv" else pd.read_parquet(manifest_path)
        manifest_hash = _hash(manifest_path)
        manifest_required = {row_id_column, "outer_role", fold_column}
        manifest_missing = sorted(manifest_required - set(manifest.columns))
        if manifest_missing:
            reasons.append(f"fold manifest lacks required columns: {manifest_missing}")
        else:
            expected = manifest.loc[manifest["outer_role"].eq(manifest_role), [row_id_column, fold_column]]
            cohort_matches = bool(np.array_equal(
                frame[row_id_column].to_numpy(), expected[row_id_column].to_numpy()
            ))
            if not cohort_matches:
                reasons.append(
                    f"prediction rows/order do not exactly match manifest role {manifest_role!r}"
                )
            if has_fold and cohort_matches:
                folds_match = bool(np.array_equal(
                    frame[fold_column].to_numpy(), expected[fold_column].to_numpy()
                ))
                if not folds_match:
                    reasons.append("prediction fold IDs do not match the canonical manifest")
    return IncumbentAudit(
        accepted=not reasons, artifact_hash=_hash(path), n_rows=int(len(frame)),
        n_groups=int(frame[group_column].nunique()), mae=mae,
        expected_mae=float(expected_mae), mae_tolerance=float(tolerance),
        has_fold_provenance=bool(has_fold), reasons=tuple(reasons),
        manifest_hash=manifest_hash,
        cohort_matches_manifest=cohort_matches,
        folds_match_manifest=folds_match,
    )
