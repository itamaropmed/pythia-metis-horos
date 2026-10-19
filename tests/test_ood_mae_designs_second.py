import numpy as np
import pytest

from horos.benchmark.ood_mae_designs import OOD_DESIGNS
from horos.benchmark.ood_mae_designs_second import (
    OOD_DESIGNS_SECOND, ood_mae_design_second,
)


def test_second_challenge_is_distinct_and_reproducible():
    assert len(OOD_DESIGNS_SECOND) == len(set(OOD_DESIGNS_SECOND)) == 10
    assert not (set(OOD_DESIGNS_SECOND) & set(OOD_DESIGNS))
    for name in OOD_DESIGNS_SECOND:
        first = ood_mae_design_second(np.random.default_rng(42), name, 128)
        second = ood_mae_design_second(np.random.default_rng(42), name, 128)
        for left, right in zip(first, second):
            assert np.array_equal(left, right)
        x, y, cap, truth = first
        assert x.shape[0] == y.shape[0] == len(truth) == 128
        assert np.all(np.isfinite(y)) and np.all(np.isfinite(truth))
        assert np.all((0 <= x) & (x <= 1)) and cap > 0


@pytest.mark.parametrize("name", OOD_DESIGNS_SECOND)
def test_second_challenge_exact_risk_matches_monte_carlo(name):
    x, y, cap, truth = ood_mae_design_second(np.random.default_rng(311), name, 100_000)
    if name == "rare_wide_outliers":
        center = x[:, 0]
    elif name == "two_scale_laplace":
        center = .5 * x[:, 0]
    elif name == "central_atom":
        center = 2 * x[:, 0]
    elif name == "symmetric_bimodal":
        center = x[:, 0]
    elif name == "skew_scale_shift":
        center = x[:, 0]
    elif name == "sparse12_uniform":
        center = 2 * x[:, 0]
    elif name == "correlated4_laplace":
        center = x[:, 0] + x[:, 1]
    elif name == "corner_beta":
        center = 2 * x[:, 0]
    elif name == "hetero_atom":
        center = x[:, 0]
    else:
        center = 3 * np.abs(x[:, 0] - .35)
    assert np.mean(np.abs(y - center)) == pytest.approx(np.mean(truth), abs=.012)
    left = np.arange(0, 1000, 2)
    right = left + 1
    distance = np.mean(np.abs(x[left] - x[right]), axis=1)
    assert np.all(np.abs(center[left] - center[right]) <= cap * distance + 1e-12)
