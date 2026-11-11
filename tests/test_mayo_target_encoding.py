import numpy as np
import pandas as pd
import pytest

from horos.models.mayo import _encode


def test_target_encoding_is_leave_one_patient_out_and_excludes_own_target():
    keys = pd.Series([f"procedure-{i}" for i in range(10)])
    y = pd.Series(np.arange(10, dtype=float) * 10.)
    groups = np.array([f"patient-{i}" for i in range(10)])
    encoded, applied = _encode(keys, y, pd.Series(["procedure-0"]), 0., groups=groups)

    # Each category is unique, so the old encoder returned y_i for row i.
    # Leave-one-patient-out makes row 0's unseen category use the other-group prior.
    assert encoded[0] == pytest.approx(y.iloc[1:].mean())
    assert encoded[0] != y.iloc[0]
    # A deployment row is legitimately encoded from all outer-fit labels.
    assert applied[0] == pytest.approx(y.iloc[0])

    changed = y.copy()
    changed.iloc[0] = 9999.
    encoded_changed, applied_changed = _encode(
        keys, changed, pd.Series(["procedure-0"]), 0., groups=groups,
    )
    assert encoded_changed[0] == pytest.approx(encoded[0])
    assert applied_changed[0] != applied[0]


def test_same_patient_is_kept_out_of_its_target_encoding_fold():
    keys = pd.Series(["shared", "shared", "shared", "other", "other", "other"])
    y = pd.Series([10., 20., 1000., 30., 40., 50.])
    groups = np.array(["p1", "p1", "p2", "p3", "p4", "p5"])
    encoded, _ = _encode(keys, y, pd.Series(["shared"]), 0., groups=groups)
    changed = y.copy()
    changed.iloc[:2] = [5000., 6000.]
    encoded_changed, _ = _encode(keys, changed, pd.Series(["shared"]), 0., groups=groups)
    # Both p1 rows are excluded together, so neither row's target enters the
    # statistics used to encode either of them.
    np.testing.assert_allclose(encoded[:2], encoded_changed[:2])


def test_target_encoding_refuses_single_independent_patient():
    with pytest.raises(ValueError, match="at least two patient groups"):
        _encode(pd.Series(["a", "b"]), pd.Series([1., 2.]),
                pd.Series(["a"]), 1., groups=np.array(["p1", "p1"]))


@pytest.mark.parametrize("groups", [["p1", None], ["p1", ""]])
def test_target_encoding_refuses_missing_patient_provenance(groups):
    with pytest.raises(ValueError, match="patient groups"):
        _encode(pd.Series(["a", "b"]), pd.Series([1., 2.]),
                pd.Series(["a"]), 1., groups=np.array(groups, dtype=object))


def test_target_encoding_refuses_nonfinite_smoothing():
    with pytest.raises(ValueError, match="smoothing"):
        _encode(pd.Series(["a", "b"]), pd.Series([1., 2.]),
                pd.Series(["a"]), np.nan, groups=np.array(["p1", "p2"]))
