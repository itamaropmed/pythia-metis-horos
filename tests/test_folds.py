import pandas as pd

from pythia.folds import build_mayo_fold_manifest


def test_fold_manifest_is_complete_and_patient_disjoint(tmp_path):
    source = tmp_path / "cases.parquet"
    output = tmp_path / "folds.parquet"
    rows = []
    for patient in range(30):
        rows.append({
            "case_durable_id": patient,
            "patient_durable_id": patient,
            "scheduled_in_room": pd.Timestamp("2025-01-01") + pd.Timedelta(days=patient),
            "case_actual_duration_time": 30.0 + patient,
        })
    # This earlier case belongs to a future holdout patient and must be excluded.
    rows.append({
        "case_durable_id": 100,
        "patient_durable_id": 29,
        "scheduled_in_room": pd.Timestamp("2025-01-03"),
        "case_actual_duration_time": 60.0,
    })
    pd.DataFrame(rows).to_parquet(source)
    report = build_mayo_fold_manifest(source, output, n_folds=3)
    manifest = pd.read_parquet(output)
    assert report.accepted
    assert report.patient_overlap_count == 0
    assert report.multi_fold_patient_count == 0
    assert report.n_excluded_patient_history == 1
    assert set(manifest.loc[manifest.outer_role.eq("development"), "fold_id"]) == {0, 1, 2}
    assert manifest.fold_id.ge(0).all()
