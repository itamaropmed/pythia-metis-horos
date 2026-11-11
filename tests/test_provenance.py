import pandas as pd

from pythia.provenance import audit_note_provenance, recut_note_bundles


def _write(tmp_path, *, dates=("2026-01-01", "2026-01-07"), folds=(0, 1)):
    bundle = tmp_path / "bundles.parquet"
    cases = tmp_path / "cases.parquet"
    pd.DataFrame({
        "case_durable_id": [1, 2],
        "note_dates_used": list(dates),
        "n_pre_cutoff_notes_used": [1, 1],
    }).to_parquet(bundle)
    pd.DataFrame({
        "case_durable_id": [1, 2],
        "patient_durable_id": [10, 20],
        "scheduled_in_room": ["2026-01-05", "2026-01-10"],
        "fold_id": list(folds),
    }).to_parquet(cases)
    return bundle, cases


def test_note_provenance_accepts_strict_cutoff_and_isolated_groups(tmp_path):
    bundle, cases = _write(tmp_path)
    result = audit_note_provenance(bundle, cases)
    assert result.accepted
    assert result.cutoff_passed
    assert result.cluster_isolation_passed


def test_note_provenance_rejects_boundary_timestamp(tmp_path):
    bundle, cases = _write(tmp_path, dates=("2026-01-03", "2026-01-07"))
    result = audit_note_provenance(bundle, cases)
    assert not result.accepted
    assert result.n_cutoff_violations == 1


def test_note_provenance_rejects_midnight_date_on_intraday_cutoff_day(tmp_path):
    bundle = tmp_path / "date-only-bundle.parquet"
    cases = tmp_path / "intraday-cases.parquet"
    pd.DataFrame({
        "case_durable_id": [1],
        # Date-only sources can be serialized as midnight timestamps.
        "note_dates_used": ["2026-01-08T00:00:00Z"],
        "n_pre_cutoff_notes_used": [1],
    }).to_parquet(bundle)
    pd.DataFrame({
        "case_durable_id": [1], "patient_durable_id": [10],
        "scheduled_in_room": ["2026-01-10T10:00:00Z"], "fold_id": [0],
    }).to_parquet(cases)

    result = audit_note_provenance(bundle, cases)

    assert not result.accepted
    assert not result.cutoff_passed
    assert result.n_cutoff_violations == 1
    assert result.n_ambiguous_date_precision_violations == 1


def test_note_provenance_accepts_precise_timestamp_before_intraday_cutoff(tmp_path):
    bundle = tmp_path / "precise-bundle.parquet"
    cases = tmp_path / "intraday-cases.parquet"
    pd.DataFrame({
        "case_durable_id": [1],
        "note_dates_used": ["2026-01-08T09:59:00Z"],
        "n_pre_cutoff_notes_used": [1],
    }).to_parquet(bundle)
    pd.DataFrame({
        "case_durable_id": [1], "patient_durable_id": [10],
        "scheduled_in_room": ["2026-01-10T10:00:00Z"], "fold_id": [0],
    }).to_parquet(cases)

    result = audit_note_provenance(bundle, cases)

    assert result.accepted
    assert result.n_ambiguous_date_precision_violations == 0


def test_note_provenance_rejects_patient_crossing_folds(tmp_path):
    bundle, cases = _write(tmp_path)
    frame = pd.read_parquet(cases)
    frame["patient_durable_id"] = [10, 10]
    frame.to_parquet(cases)
    result = audit_note_provenance(bundle, cases)
    assert not result.accepted
    assert result.cluster_isolation_passed is False


def test_recut_removes_late_notes_and_reaudits(tmp_path):
    bundle = tmp_path / "source.parquet"
    cases = tmp_path / "cases.parquet"
    output = tmp_path / "recut.parquet"
    pd.DataFrame({
        "case_durable_id": [1],
        "note_text": ["[NOTE_TYPE: OLD] old [NOTE_TYPE: LATE] late"],
        "n_pre_cutoff_notes_used": [2],
        "note_dates_used": ["2026-01-01; 2026-01-04"],
        "note_doc_types_used": ["OLD; LATE"],
    }).to_parquet(bundle)
    pd.DataFrame({
        "case_durable_id": [1], "patient_durable_id": [10],
        "scheduled_in_room": ["2026-01-05"], "fold_id": [0],
    }).to_parquet(cases)
    report = recut_note_bundles(bundle, cases, output, history_floor_days=None)
    assert report.n_retained_notes == 1
    assert audit_note_provenance(output, cases).accepted


def test_recut_removes_midnight_note_on_intraday_cutoff_calendar_day(tmp_path):
    bundle = tmp_path / "source.parquet"
    cases = tmp_path / "cases.parquet"
    output = tmp_path / "recut.parquet"
    pd.DataFrame({
        "case_durable_id": [1],
        "note_text": "[NOTE_TYPE: SAFE] old [NOTE_TYPE: AMBIGUOUS] boundary",
        "n_pre_cutoff_notes_used": [2],
        "note_dates_used": "2026-01-01T00:00:00Z; 2026-01-08T00:00:00Z",
        "note_doc_types_used": "SAFE; AMBIGUOUS",
    }).to_parquet(bundle)
    pd.DataFrame({
        "case_durable_id": [1], "patient_durable_id": [10],
        "scheduled_in_room": ["2026-01-10T10:00:00Z"], "fold_id": [0],
    }).to_parquet(cases)

    report = recut_note_bundles(bundle, cases, output, history_floor_days=None)

    assert report.n_retained_notes == 1
    stored = pd.read_parquet(output).iloc[0]
    assert "SAFE" in stored["note_text"]
    assert "AMBIGUOUS" not in stored["note_text"]
    assert audit_note_provenance(output, cases).accepted
