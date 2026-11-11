"""Aggregate-only audit of Mayo note availability and saved-bundle cutoffs.

The auditor reads note IDs and timestamps only; it never loads note text and
prints no case or patient identifiers. The default rule matches the study
specification: date-only note_date < scheduled_in_room - 2 days, with a
548-day lookback from scheduled_in_room.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _day(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def audit_frames(
    cases: pd.DataFrame,
    note_rows: pd.DataFrame,
    bundles: pd.DataFrame | None = None,
    *,
    cutoff_days: int = 2,
    lookback_days: int = 548,
) -> dict[str, object]:
    """Summarize date eligibility without reading or returning clinical text."""
    required_cases = {"case_durable_id", "patient_durable_id", "scheduled_in_room"}
    required_notes = {"epic_id", "doc_date"}
    if not required_cases.issubset(cases.columns) or not required_notes.issubset(note_rows.columns):
        raise ValueError("cases/notes are missing required ID or date columns")
    if cutoff_days < 0 or lookback_days <= cutoff_days:
        raise ValueError("lookback_days must exceed nonnegative cutoff_days")
    if cases["case_durable_id"].duplicated().any():
        raise ValueError("case IDs must be unique")

    c = cases[["case_durable_id", "patient_durable_id", "scheduled_in_room"]].copy()
    c["patient"] = c["patient_durable_id"].astype("string")
    c["sched_day"] = _day(c["scheduled_in_room"])
    c["sched_ts"] = pd.to_datetime(c["scheduled_in_room"], errors="coerce", utc=True)
    n = note_rows[["epic_id", "doc_date"]].copy()
    n["patient"] = n["epic_id"].astype("string")
    parsed = pd.to_datetime(n["doc_date"], errors="coerce", utc=True)
    n["note_day"] = parsed.dt.tz_localize(None).dt.normalize()
    nonmidnight = parsed.notna() & (
        parsed.dt.hour.ne(0) | parsed.dt.minute.ne(0) | parsed.dt.second.ne(0)
    )
    grouped = {
        str(patient): np.sort(group.dropna().to_numpy(dtype="datetime64[ns]"))
        for patient, group in n.dropna(subset=["patient"]).groupby("patient", sort=False)["note_day"]
    }
    eligible_counts = np.zeros(len(c), dtype=np.int64)
    cutoff_day_ambiguous_pairs = np.zeros(len(c), dtype=np.int64)
    exact_timestamp_accepted_ambiguous_pairs = np.zeros(len(c), dtype=np.int64)
    missing_patient_history = 0
    schedules = c["sched_day"].to_numpy(dtype="datetime64[ns]")
    schedule_timestamps = c["sched_ts"].to_numpy(dtype="datetime64[ns]")
    for i, (patient, sched, sched_ts) in enumerate(zip(c["patient"], schedules, schedule_timestamps)):
        dates = grouped.get(str(patient))
        if dates is None or np.isnat(sched):
            missing_patient_history += 1
            continue
        hi = sched - np.timedelta64(cutoff_days, "D")
        lo = sched - np.timedelta64(lookback_days, "D")
        eligible_counts[i] = np.searchsorted(dates, hi, side="left") - np.searchsorted(dates, lo, side="left")
        # A date-only note at midnight cannot establish that it existed before
        # an intraday cutoff on that same calendar date. Count these separately
        # from definitely eligible dates; the old timestamp-exact check
        # incorrectly accepted them whenever scheduled_in_room had a time.
        cutoff_day_ambiguous_pairs[i] = (
            np.searchsorted(dates, hi + np.timedelta64(1, "D"), side="left")
            - np.searchsorted(dates, hi, side="left")
        )
        if not np.isnat(sched_ts):
            exact_cutoff = sched_ts - np.timedelta64(cutoff_days, "D")
            midnight_note = hi.astype("datetime64[ns]")
            if midnight_note < exact_cutoff:
                exact_timestamp_accepted_ambiguous_pairs[i] = cutoff_day_ambiguous_pairs[i]

    result: dict[str, object] = {
        "cases": int(len(c)),
        "note_rows": int(len(n)),
        "unique_patients_with_note_rows": int(len(grouped)),
        "note_timestamps_nonmidnight": int(nonmidnight.sum()),
        "cases_without_patient_note_history": int(missing_patient_history),
        "eligible_note_rows_scheduled_cutoff_lookback": int(eligible_counts.sum()),
        "cases_with_at_least_one_eligible_note": int((eligible_counts > 0).sum()),
        "eligible_case_coverage": float((eligible_counts > 0).mean()) if len(c) else 0.0,
        "mean_eligible_notes_per_case": float(eligible_counts.mean()) if len(c) else 0.0,
        "eligible_note_slots_if_capped_at_12": int(np.minimum(eligible_counts, 12).sum()),
        "ambiguous_date_only_note_case_pairs_on_cutoff_calendar_day": int(cutoff_day_ambiguous_pairs.sum()),
        "cases_with_ambiguous_date_only_note_on_cutoff_calendar_day": int((cutoff_day_ambiguous_pairs > 0).sum()),
        "ambiguous_pairs_that_midnight_timestamp_check_would_accept": int(
            exact_timestamp_accepted_ambiguous_pairs.sum()
        ),
        "cases_with_ambiguous_pair_accepted_by_midnight_timestamp_check": int(
            (exact_timestamp_accepted_ambiguous_pairs > 0).sum()
        ),
        "cutoff_days": int(cutoff_days),
        "lookback_days": int(lookback_days),
    }
    if bundles is None:
        return result

    required_bundles = {"case_durable_id", "note_dates_used", "n_pre_cutoff_notes_used"}
    if not required_bundles.issubset(bundles.columns):
        raise ValueError("bundles are missing date-list/count columns")
    if bundles["case_durable_id"].duplicated().any():
        raise ValueError("bundle case IDs must be unique")
    joined = c[["case_durable_id", "sched_day"]].merge(
        bundles[["case_durable_id", "note_dates_used", "n_pre_cutoff_notes_used"]],
        on="case_durable_id", how="right", validate="one_to_one", indicator=True,
    )
    date_total = bad_date_tokens = count_mismatches = recent = old = post_day = same_day = 0
    valid_count = valid_cases = recent_cases = old_cases = 0
    for _, sched_day, raw_value, stored_count, match_status in joined.itertuples(index=False, name=None):
        raw = "" if pd.isna(raw_value) else str(raw_value)
        tokens = [token.strip() for token in raw.split(";") if token.strip()]
        dates = pd.to_datetime(pd.Series(tokens), errors="coerce", utc=True)
        bad_date_tokens += int(dates.isna().sum())
        dates = dates.dropna().dt.tz_localize(None).dt.normalize()
        date_total += len(dates)
        expected = 0 if pd.isna(stored_count) else int(stored_count)
        count_mismatches += int(expected != len(tokens))
        if match_status != "both" or pd.isna(sched_day):
            continue
        delta = (sched_day - dates).dt.days
        recent_mask = delta <= cutoff_days
        old_mask = delta > lookback_days
        valid_mask = (delta > cutoff_days) & (delta <= lookback_days)
        recent += int(recent_mask.sum()); old += int(old_mask.sum())
        post_day += int((delta < 0).sum()); same_day += int((delta == 0).sum())
        valid_count += int(valid_mask.sum())
        valid_cases += int(valid_mask.any())
        recent_cases += int(recent_mask.any())
        old_cases += int(old_mask.any())
    result["bundle"] = {
        "rows": int(len(bundles)),
        "unmatched_case_ids": int((joined["_merge"] == "right_only").sum()),
        "stored_date_tokens": int(date_total),
        "unparseable_date_tokens": int(bad_date_tokens),
        "stored_count_mismatches": int(count_mismatches),
        "dates_on_or_after_required_cutoff": int(recent),
        "cases_with_date_on_or_after_required_cutoff": int(recent_cases),
        "dates_older_than_required_lookback": int(old),
        "cases_with_date_older_than_required_lookback": int(old_cases),
        "dates_after_scheduled_calendar_day": int(post_day),
        "same_calendar_day_dates": int(same_day),
        "date_tokens_valid_under_required_window": int(valid_count),
        "cases_with_at_least_one_stored_valid_date": int(valid_cases),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="tabular parquet path")
    parser.add_argument("--notes", required=True, type=Path, help="notes parquet path")
    parser.add_argument("--bundles", type=Path, help="optional bundle parquet path")
    parser.add_argument("--cutoff-days", type=int, default=2)
    parser.add_argument("--lookback-days", type=int, default=548)
    args = parser.parse_args()
    cases = pd.read_parquet(args.cases, columns=["case_durable_id", "patient_durable_id", "scheduled_in_room"])
    note_rows = pd.read_parquet(args.notes, columns=["epic_id", "doc_date"])
    bundles = None
    if args.bundles:
        bundles = pd.read_parquet(args.bundles, columns=[
            "case_durable_id", "note_dates_used", "n_pre_cutoff_notes_used",
        ])
    print(json.dumps(audit_frames(
        cases, note_rows, bundles,
        cutoff_days=args.cutoff_days, lookback_days=args.lookback_days,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
