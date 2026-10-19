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
        target_column="y", group_column="entity",
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
            target_column="y", group_column="entity",
        )
