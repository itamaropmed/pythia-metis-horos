import numpy as np

from research.instrumented_20261019.split_safety import center_by_training_groups


def test_heldout_outcomes_cannot_change_training_residualization():
    groups = np.array(["a"] * 4 + ["b"] * 4)
    train = np.array([True, True, False, False] * 2)
    values = np.array([1., 3., 8., 9., 2., 4., 7., 11.])
    changed = values.copy()
    changed[~train] += 1000.

    first = center_by_training_groups(values, groups, train)
    second = center_by_training_groups(changed, groups, train)

    np.testing.assert_array_equal(first[train], second[train])
    np.testing.assert_allclose(first[train].reshape(2, 2).mean(axis=1), 0.)


def test_unseen_group_uses_training_only_global_fallback():
    values = np.array([[1.], [3.], [100.]])
    groups = np.array(["seen", "seen", "unseen"])
    train = np.array([True, True, False])
    transformed = center_by_training_groups(values, groups, train)
    assert transformed[2, 0] == 98.


def test_calibration_and_test_outcomes_cannot_change_fit_trained_reader_scores():
    rng = np.random.default_rng(20261024)
    groups = np.repeat(["a", "b", "c"], 100)
    split = np.tile(np.repeat(["fit", "cal", "test"], [50, 25, 25]), 3)
    train, heldout = split == "fit", split != "fit"
    X = rng.normal(size=(len(groups), 4))
    y = X[:, 0] * 2 + rng.normal(size=len(groups))

    def trained_scores(target):
        Xc = center_by_training_groups(X, groups, train)
        yc = center_by_training_groups(target[:, None], groups, train)[:, 0]
        scale = Xc[train].std(0) + 1e-9
        Xs = Xc / scale
        beta = np.linalg.solve(Xs[train].T @ Xs[train] + np.eye(X.shape[1]),
                               Xs[train].T @ yc[train])
        return Xs[~train] @ beta

    baseline = trained_scores(y)
    changed = y.copy()
    changed[heldout] += 100.
    np.testing.assert_allclose(trained_scores(changed), baseline, rtol=0., atol=0.)
