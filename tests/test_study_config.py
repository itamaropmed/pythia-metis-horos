from pathlib import Path

import pandas as pd

from horos.config import load_study
from horos.pipeline import prepare_study


def test_generic_study_preparation(tmp_path: Path):
    data = tmp_path / "events.parquet"
    manifest = tmp_path / "manifest.parquet"
    config = tmp_path / "study.yaml"
    pd.DataFrame({
        "event": range(20), "subject": range(20),
        "when": pd.date_range("2025-01-01", periods=20),
        "outcome": range(1, 21),
    }).to_parquet(data)
    config.write_text(
        "name: generic-test\n"
        "loss: mae\n"
        "dataset:\n"
        "  path: events.parquet\n"
        "  row_id: event\n"
        "  target: outcome\n"
        "  group: subject\n"
        "  time: when\n"
        "split:\n"
        "  kind: grouped_temporal_holdout\n"
        "  test_fraction: 0.20\n"
        "  inner_folds: 3\n"
        "  manifest: manifest.parquet\n"
    )
    study = load_study(config)
    result = prepare_study(config)
    assert study.dataset.row_id == "event"
    assert result["status"] == "prepared"
    assert manifest.exists()
