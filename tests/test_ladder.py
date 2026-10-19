import numpy as np
import pytest

from pythia import FoldMatrix, Representation, TaskCard, evaluate_ladder


def _folds(y, offset=0.0):
    y = np.asarray(y, float)
    p = np.vstack([y + offset + 0.2, y + offset - 0.2])
    return FoldMatrix(y, p, ("a", "b"), ("x", "y"), np.arange(y.size) % 2,
                      row_ids=np.arange(y.size))


def test_ladder_requires_aligned_rows():
    with pytest.raises(ValueError):
        evaluate_ladder(TaskCard("y", "mse"), [
            (Representation("a"), _folds([1, 2, 3])),
            (Representation("b"), _folds([1, 2, 4])),
        ])


def test_ladder_reports_changes_without_causal_harm_claim():
    result = evaluate_ladder(TaskCard("y", "mse"), [
        (Representation("a"), _folds([1, 2, 3], 0.2)),
        (Representation("b"), _folds([1, 2, 3], 0.0)),
    ])
    assert len(result["entries"]) == 2
    assert "does not imply" in result["entries"][1]["change_from_previous"]["interpretation"]

