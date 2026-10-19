import pandas as pd

from pythia.incumbent import audit_incumbent


def test_incumbent_requires_fold_provenance(tmp_path):
    path = tmp_path / "pred.csv"
    pd.DataFrame({
        "case": [1, 2], "patient": [10, 20], "y": [10.0, 20.0],
        "pred": [9.0, 19.0],
    }).to_csv(path, index=False)
    result = audit_incumbent(
        path, target_column="y", prediction_column="pred", row_id_column="case",
        group_column="patient", expected_mae=1.0,
    )
    assert not result.accepted
    assert not result.has_fold_provenance


def test_incumbent_accepts_exact_oof_artifact(tmp_path):
    path = tmp_path / "pred.csv"
    pd.DataFrame({
        "case": [1, 2], "patient": [10, 20], "fold_id": [0, 1],
        "y": [10.0, 20.0], "pred": [9.0, 19.0],
    }).to_csv(path, index=False)
    result = audit_incumbent(
        path, target_column="y", prediction_column="pred", row_id_column="case",
        group_column="patient", expected_mae=1.0,
    )
    assert result.accepted and result.has_fold_provenance


def test_incumbent_requires_exact_manifest_rows_and_folds(tmp_path):
    path = tmp_path / "pred.csv"
    manifest = tmp_path / "folds.parquet"
    pd.DataFrame({
        "case": [1, 2], "patient": [10, 20], "fold_id": [0, 1],
        "y": [10.0, 20.0], "pred": [9.0, 19.0],
    }).to_csv(path, index=False)
    pd.DataFrame({
        "case": [1, 2, 3], "outer_role": ["development", "development", "holdout"],
        "fold_id": [0, 1, 2],
    }).to_parquet(manifest)
    result = audit_incumbent(
        path, target_column="y", prediction_column="pred", row_id_column="case",
        group_column="patient", expected_mae=1.0, manifest_path=manifest,
    )
    assert result.accepted
    assert result.cohort_matches_manifest and result.folds_match_manifest

    swapped = pd.read_csv(path).iloc[::-1]
    swapped.to_csv(path, index=False)
    rejected = audit_incumbent(
        path, target_column="y", prediction_column="pred", row_id_column="case",
        group_column="patient", expected_mae=1.0, manifest_path=manifest,
    )
    assert not rejected.accepted
    assert rejected.cohort_matches_manifest is False
