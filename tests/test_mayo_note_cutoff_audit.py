import pandas as pd
import pytest

from scripts.audit_mayo_note_cutoff import audit_frames


def test_audit_counts_exact_schedule_cutoff_and_bundle_violations():
    cases = pd.DataFrame({
        "case_durable_id": [1, 2],
        "patient_durable_id": ["p1", "p2"],
        "scheduled_in_room": ["2026-01-10T10:00:00Z", "2026-01-10T10:00:00Z"],
    })
    notes = pd.DataFrame({
        "epic_id": ["p1"] * 4,
        "doc_date": ["2026-01-05", "2026-01-07", "2026-01-08", "2026-01-10"],
    })
    bundles = pd.DataFrame({
        "case_durable_id": [1, 2],
        "note_dates_used": ["2026-01-07; 2026-01-08; 2026-01-10", ""],
        "n_pre_cutoff_notes_used": [3, 0],
    })
    result = audit_frames(cases, notes, bundles, cutoff_days=2, lookback_days=5)
    assert result["eligible_note_rows_scheduled_cutoff_lookback"] == 2
    assert result["cases_with_at_least_one_eligible_note"] == 1
    assert result["bundle"]["dates_on_or_after_required_cutoff"] == 2
    assert result["bundle"]["date_tokens_valid_under_required_window"] == 1
    assert result["bundle"]["same_calendar_day_dates"] == 1
    assert result["bundle"]["stored_count_mismatches"] == 0
    assert result["ambiguous_date_only_note_case_pairs_on_cutoff_calendar_day"] == 1
    assert result["cases_with_ambiguous_date_only_note_on_cutoff_calendar_day"] == 1
    assert result["ambiguous_pairs_that_midnight_timestamp_check_would_accept"] == 1


def test_audit_does_not_treat_midnight_scheduled_cutoff_as_intraday_proof():
    cases = pd.DataFrame({
        "case_durable_id": [1],
        "patient_durable_id": ["p1"],
        "scheduled_in_room": ["2026-01-10T00:00:00Z"],
    })
    notes = pd.DataFrame({"epic_id": ["p1"], "doc_date": ["2026-01-08"]})
    result = audit_frames(cases, notes, cutoff_days=2, lookback_days=5)
    assert result["ambiguous_date_only_note_case_pairs_on_cutoff_calendar_day"] == 1
    assert result["ambiguous_pairs_that_midnight_timestamp_check_would_accept"] == 0


def test_audit_refuses_duplicate_case_ids():
    cases = pd.DataFrame({
        "case_durable_id": [1, 1],
        "patient_durable_id": ["p1", "p1"],
        "scheduled_in_room": ["2026-01-10", "2026-01-11"],
    })
    notes = pd.DataFrame({"epic_id": ["p1"], "doc_date": ["2026-01-01"]})
    with pytest.raises(ValueError, match="case IDs must be unique"):
        audit_frames(cases, notes)
