import numpy as np
import pandas as pd
import pytest

from horos.models.crossfit import run_crossfit


def test_generic_crossfit_is_complete():
    frame = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0], "entity": [10, 20, 30, 40],
        "fold_id": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 4.0],
    })
    result = run_crossfit(
        frame, lambda fit, val, seed: val["x"].to_numpy(),
        target_column="y", group_column="entity", feature_columns=["x"],
    )
    assert np.array_equal(result.predictions, frame.y)
    assert result.fold_losses == (0.0, 0.0)


def test_generic_crossfit_refuses_entity_leakage():
    frame = pd.DataFrame({
        "y": [1.0, 2.0], "entity": [10, 10], "fold_id": [0, 1], "x": [1.0, 2.0],
    })
    with pytest.raises(RuntimeError, match="leakage"):
        run_crossfit(
            frame, lambda fit, val, seed: val["x"].to_numpy(),
            target_column="y", group_column="entity", feature_columns=["x"],
        )


def test_fold_predictor_never_receives_validation_outcomes_or_identity_columns():
    frame = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0], "entity": [10, 20, 30, 40],
        "fold_id": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 4.0],
    })

    def predictor(fit, val, seed):
        assert list(val.columns) == ["x"]
        assert "y" in fit.columns
        return val["x"].to_numpy()

    result = run_crossfit(
        frame, predictor, target_column="y", group_column="entity", feature_columns=["x"],
    )
    assert np.isfinite(result.predictions).all()


def test_validation_target_access_fails_closed():
    frame = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0], "entity": [10, 20, 30, 40],
        "fold_id": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 4.0],
    })

    def leaking_predictor(fit, val, seed):
        return val["y"].to_numpy()

    with pytest.raises(KeyError, match="y"):
        run_crossfit(
            frame, leaking_predictor, target_column="y", group_column="entity",
            feature_columns=["x"],
        )


def test_feature_allowlist_excludes_accidental_outcome_proxy():
    frame = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0], "entity": [10, 20, 30, 40],
        "fold_id": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 4.0],
        "post_outcome_duration_copy": [1.0, 2.0, 3.0, 4.0],
    })
    seen_columns = []

    def predictor(fit, val, seed):
        seen_columns.append(tuple(val.columns))
        return val["x"].to_numpy()

    run_crossfit(
        frame, predictor, target_column="y", group_column="entity", feature_columns=["x"],
    )
    assert seen_columns == [("x",), ("x",)]


@pytest.mark.parametrize("features", [["x", "x"], ["y"], ["missing"]])
def test_feature_allowlist_must_be_valid(features):
    frame = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0], "entity": [10, 20, 30, 40],
        "fold_id": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError):
        run_crossfit(
            frame, lambda fit, val, seed: val.iloc[:, 0].to_numpy(),
            target_column="y", group_column="entity", feature_columns=features,
        )
