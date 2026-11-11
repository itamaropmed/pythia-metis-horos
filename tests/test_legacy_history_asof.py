import numpy as np
import pandas as pd
import pytest

from scripts.legacy_mayo.build import ACTUAL_OUT, frame as build_feature_frame, history


def cases():
    return pd.DataFrame({
        "case_durable_id": ["a", "b", "c"],
        "patient_durable_id": ["p1", "p2", "p3"],
        "surgeon_durable_id": ["s1", "s2", "s3"],
        "primary_procedure_name": ["proc", "proc", "proc"],
        "scheduled_in_room": pd.to_datetime([
            "2020-01-01T08:00:00Z", "2020-01-01T09:00:00Z", "2020-01-01T13:00:00Z",
        ]),
        "case_actual_patient_in_room": pd.to_datetime([
            "2020-01-01T08:00:00Z", "2020-01-01T09:00:00Z", "2020-01-01T12:00:00Z",
        ]),
        ACTUAL_OUT: pd.to_datetime([
            "2020-01-01T11:00:00Z", "2020-01-01T10:00:00Z", "2020-01-01T12:30:00Z",
        ]),
        "case_actual_duration_time": [180., 100., 30.],
    })


def test_history_uses_only_cases_completed_before_scheduled_time():
    result = history(cases())
    row_b = result.set_index("case_durable_id").loc["b"]
    row_c = result.set_index("case_durable_id").loc["c"]

    # Case A started first but was still running at B's scheduled time.
    assert row_b.hist_proc_count == 0
    assert np.isnan(row_b.hist_proc_mean)
    # By C's scheduled time, A and B are complete and usable. C's own actual
    # completion timestamp is also earlier than its malformed schedule, so an
    # explicit same-case exclusion is required to prevent self-target leakage.
    assert row_c.hist_proc_count == 2
    assert row_c.hist_proc_mean == 140.
    assert row_c.hist_proc_median == 140.


def test_completion_time_is_never_a_model_feature():
    features, _ = build_feature_frame(history(cases()))
    assert ACTUAL_OUT not in features.columns
    assert "case_actual_duration_time" not in features.columns


def test_history_refuses_missing_or_invalid_completion_times():
    with pytest.raises(ValueError, match="require completion timestamp"):
        history(cases().drop(columns=[ACTUAL_OUT]))
    broken = cases()
    broken.loc[1, ACTUAL_OUT] = pd.NaT
    with pytest.raises(ValueError, match="timestamps must be complete"):
        history(broken)
