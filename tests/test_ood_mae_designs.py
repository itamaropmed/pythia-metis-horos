import numpy as np

from horos.benchmark.ood_mae_designs import (
    OOD_DESIGNS, ood_conditional_median, ood_mae_design,
)


def test_ten_ood_designs_have_valid_shapes_and_positive_known_risk():
    assert len(OOD_DESIGNS) == 10
    for name in OOD_DESIGNS:
        x, y, l, truth = ood_mae_design(np.random.default_rng(7), name, 32)
        assert x.shape[0] == y.size == truth.size == 32
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert np.all((0 <= x) & (x <= 1))
        assert np.all(truth > 0)
        assert l > 0


def test_ood_analytic_mae_truth_and_lipschitz_caps_against_simulation():
    for design_id, name in enumerate(OOD_DESIGNS):
        x, y, l, truth = ood_mae_design(
            np.random.default_rng(200 + design_id), name, 20_000,
        )
        median = ood_conditional_median(name, x)
        assert abs(np.mean(np.abs(y - median)) - np.mean(truth)) < .02
        left = np.arange(1000)
        right = np.arange(1000, 2000)
        distance = np.mean(np.abs(x[left] - x[right]), axis=1)
        assert np.all(np.abs(median[left] - median[right]) <= l * distance + 1e-12)
