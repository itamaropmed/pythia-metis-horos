"""Privacy-preserving audits for case-level clinical-note bundles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class NoteProvenanceAudit:
    accepted: bool
    bundle_hash: str
    case_table_hash: str
    n_bundle_rows: int
    n_case_rows: int
    n_cases_with_notes: int
    n_notes_declared: int
    n_note_dates_parsed: int
    n_cutoff_violations: int
    n_count_mismatches: int
    n_unmatched_bundle_cases: int
    cutoff_days: int
    cutoff_passed: bool
    cluster_isolation_passed: bool | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NoteRecutReport:
    output_path: str
    source_hash: str
    case_table_hash: str
    output_hash: str
    n_source_cases: int
    n_output_cases: int
    n_source_notes: int
    n_retained_notes: int
    n_cases_with_retained_notes: int
    cutoff_days: int
    history_floor_days: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns)
    return pd.read_parquet(path, columns=columns)


def audit_note_provenance(
    bundle_path: str | Path,
    case_path: str | Path,
    *,
    case_id_column: str = "case_durable_id",
    patient_column: str = "patient_durable_id",
    scheduled_column: str = "scheduled_in_room",
    note_dates_column: str = "note_dates_used",
    note_count_column: str = "n_pre_cutoff_notes_used",
    fold_column: str = "fold_id",
    cutoff_days: int = 2,
    require_cluster_isolation: bool = True,
) -> NoteProvenanceAudit:
    """Audit cutoff and patient-fold isolation without loading clinical note text.

    The bundle is expected to contain semicolon-delimited note timestamps.  The
    returned object contains only aggregate counts and file hashes; no case,
    patient, or note values are exposed.
    """
    bundle_path, case_path = Path(bundle_path), Path(case_path)
    bundle_cols = [case_id_column, note_dates_column, note_count_column]
    cases_available = set(pd.read_parquet(case_path).columns) if case_path.suffix.lower() != ".csv" else set(pd.read_csv(case_path, nrows=0).columns)
    case_cols = [case_id_column, patient_column, scheduled_column]
    if fold_column in cases_available:
        case_cols.append(fold_column)
    bundle = _read(bundle_path, bundle_cols)
    cases = _read(case_path, case_cols)
    reasons: list[str] = []

    if bundle[case_id_column].isna().any() or bundle[case_id_column].duplicated().any():
        reasons.append("bundle case IDs are null or non-unique")
    if cases[case_id_column].isna().any() or cases[case_id_column].duplicated().any():
        reasons.append("case-table case IDs are null or non-unique")

    dates = bundle[note_dates_column].fillna("").astype(str)
    declared = pd.to_numeric(bundle[note_count_column], errors="coerce")
    split_dates = dates.map(lambda value: [part.strip() for part in value.split(";") if part.strip()])
    observed_counts = split_dates.map(len)
    count_mismatch = declared.isna() | declared.ne(observed_counts)
    n_count_mismatches = int(count_mismatch.sum())
    if n_count_mismatches:
        reasons.append(f"{n_count_mismatches} bundle rows disagree with their declared note count")

    compact = bundle[[case_id_column]].copy()
    compact["_dates"] = split_dates
    notes = compact.explode("_dates", ignore_index=True)
    notes = notes[notes["_dates"].notna() & notes["_dates"].ne("")]
    notes["_note_time"] = pd.to_datetime(notes["_dates"], errors="coerce", utc=True)
    n_unparsed = int(notes["_note_time"].isna().sum())
    if n_unparsed:
        reasons.append(f"{n_unparsed} note timestamps could not be parsed")

    case_times = cases[[case_id_column, scheduled_column]].copy()
    case_times["_scheduled"] = pd.to_datetime(case_times[scheduled_column], errors="coerce", utc=True)
    n_bad_scheduled = int(case_times["_scheduled"].isna().sum())
    if n_bad_scheduled:
        reasons.append(f"{n_bad_scheduled} scheduled timestamps could not be parsed")
    joined = notes.merge(case_times[[case_id_column, "_scheduled"]], on=case_id_column, how="left", indicator=True)
    n_unmatched = int(joined.loc[joined["_merge"].ne("both"), case_id_column].nunique())
    if n_unmatched:
        reasons.append(f"{n_unmatched} bundle cases are absent from the case table")
    valid = joined["_note_time"].notna() & joined["_scheduled"].notna()
    threshold = joined["_scheduled"] - pd.Timedelta(days=cutoff_days)
    # The frozen rule is strict: note_date < scheduled_in_room - 2 days.
    n_cutoff_violations = int((valid & joined["_note_time"].ge(threshold)).sum())
    cutoff_passed = not (n_unparsed or n_bad_scheduled or n_unmatched or n_count_mismatches or n_cutoff_violations)
    if n_cutoff_violations:
        reasons.append(f"{n_cutoff_violations} notes violate the strict pre-cutoff rule")

    cluster_isolation: bool | None
    if fold_column not in cases:
        cluster_isolation = None
        if require_cluster_isolation:
            reasons.append(f"case table lacks {fold_column!r}; patient-cluster isolation is unverified")
    else:
        fold_counts = cases.dropna(subset=[patient_column, fold_column]).groupby(patient_column)[fold_column].nunique()
        leaked_patients = int(fold_counts.gt(1).sum())
        cluster_isolation = bool(leaked_patients == 0 and cases[fold_column].notna().all())
        if leaked_patients:
            reasons.append(f"{leaked_patients} patients occur in more than one fold")
        if cases[fold_column].isna().any():
            reasons.append("fold assignments are incomplete")

    return NoteProvenanceAudit(
        accepted=cutoff_passed and (cluster_isolation is True or not require_cluster_isolation),
        bundle_hash=_hash(bundle_path),
        case_table_hash=_hash(case_path),
        n_bundle_rows=int(len(bundle)),
        n_case_rows=int(len(cases)),
        n_cases_with_notes=int(observed_counts.gt(0).sum()),
        n_notes_declared=int(declared.fillna(0).sum()),
        n_note_dates_parsed=int(notes["_note_time"].notna().sum()),
        n_cutoff_violations=n_cutoff_violations,
        n_count_mismatches=n_count_mismatches,
        n_unmatched_bundle_cases=n_unmatched,
        cutoff_days=int(cutoff_days),
        cutoff_passed=cutoff_passed,
        cluster_isolation_passed=cluster_isolation,
        reasons=tuple(reasons),
    )


_NOTE_SPLIT = re.compile(r"(?=\[NOTE_TYPE:)")


def recut_note_bundles(
    bundle_path: str | Path,
    case_path: str | Path,
    output_path: str | Path,
    *,
    case_id_column: str = "case_durable_id",
    scheduled_column: str = "scheduled_in_room",
    cutoff_days: int = 2,
    history_floor_days: int | None = 548,
) -> NoteRecutReport:
    """Re-cut an existing capped bundle to a strict deployable timestamp rule.

    This cannot recover older notes discarded by the source bundle's cap. A
    full raw-corpus rebuild is still required for the final information rung.
    """
    bundle_path, case_path, output_path = Path(bundle_path), Path(case_path), Path(output_path)
    bundle = _read(bundle_path, [
        case_id_column, "note_text", "note_dates_used", "note_doc_types_used",
        "n_pre_cutoff_notes_used",
    ])
    cases = _read(case_path, [case_id_column, scheduled_column])
    if bundle[case_id_column].duplicated().any() or cases[case_id_column].duplicated().any():
        raise ValueError("case IDs must be unique before re-cutting")
    joined = cases.merge(bundle, on=case_id_column, how="inner", validate="one_to_one")
    scheduled = pd.to_datetime(joined[scheduled_column], errors="coerce", utc=True)
    if scheduled.isna().any():
        raise ValueError("scheduled timestamps must be complete and parseable")

    rows: list[dict] = []
    source_notes = 0
    retained_notes = 0
    for pos, row in enumerate(joined.itertuples(index=False)):
        case_id = getattr(row, case_id_column)
        text = str(getattr(row, "note_text") or "")
        segments = [part for part in _NOTE_SPLIT.split(text) if part.strip()]
        dates = [part.strip() for part in str(getattr(row, "note_dates_used") or "").split(";") if part.strip()]
        types = [part.strip() for part in str(getattr(row, "note_doc_types_used") or "").split(";") if part.strip()]
        declared = int(getattr(row, "n_pre_cutoff_notes_used"))
        if not (len(segments) == len(dates) == len(types) == declared):
            raise ValueError(f"bundle metadata is misaligned at joined row {pos}")
        timestamps = pd.to_datetime(pd.Series(dates), errors="coerce", utc=True)
        if timestamps.isna().any():
            raise ValueError(f"unparseable note timestamp at joined row {pos}")
        cutoff = scheduled.iloc[pos] - pd.Timedelta(days=cutoff_days)
        keep = timestamps.lt(cutoff)
        if history_floor_days is not None:
            keep &= timestamps.ge(scheduled.iloc[pos] - pd.Timedelta(days=history_floor_days))
        selected = [idx for idx, flag in enumerate(keep.tolist()) if flag]
        source_notes += len(segments)
        retained_notes += len(selected)
        selected_text = " ".join(segments[idx] for idx in selected)
        rows.append({
            case_id_column: case_id,
            "note_text": selected_text,
            "n_pre_cutoff_notes_used": len(selected),
            "note_doc_types_used": "; ".join(types[idx] for idx in selected),
            "note_dates_used": "; ".join(dates[idx] for idx in selected),
            "note_chars": len(selected_text),
        })
    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return NoteRecutReport(
        output_path=str(output_path), source_hash=_hash(bundle_path),
        case_table_hash=_hash(case_path), output_hash=_hash(output_path),
        n_source_cases=int(len(bundle)), n_output_cases=int(len(output)),
        n_source_notes=int(source_notes), n_retained_notes=int(retained_notes),
        n_cases_with_retained_notes=int(output["n_pre_cutoff_notes_used"].gt(0).sum()),
        cutoff_days=int(cutoff_days), history_floor_days=history_floor_days,
    )
