import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

from horos.models.temporal_splits import forward_group_splits


def test_forward_splits_are_time_ordered_and_entity_disjoint():
    # 30 independent patients per date, with duplicate rows per patient.
    entities = np.repeat(np.arange(120).astype(str), 2)
    times = np.repeat(np.repeat(pd.date_range("2020-01-01", periods=4, freq="D"), 30), 2)
    splits = forward_group_splits(times, entities, n_splits=3, min_train_fraction=.25)
    assert len(splits) == 3
    covered = []
    for train, valid in splits:
        assert times[train].max() < times[valid].min()
        assert set(entities[train]).isdisjoint(entities[valid])
        covered.extend(valid.tolist())
    assert len(covered) == len(set(covered))


def test_repeated_patient_history_is_removed_from_each_validation_fold():
    # Ten patients recur from the first into the third period; enough other
    # patients remain in the prefix to fit each forward fold.
    entities, times = [], []
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    for period, date in enumerate(dates):
        entities.extend([f"returning-{i}" for i in range(10)] if period in (0, 2) else [])
        entities.extend([f"new-{period}-{i}" for i in range(20)])
        times.extend([date] * (30 if period in (0, 2) else 20))
    entities, times = np.asarray(entities), np.asarray(times)
    splits = forward_group_splits(times, entities, n_splits=3, min_train_fraction=.2)
    assert all(set(entities[tr]).isdisjoint(entities[va]) for tr, va in splits)
    assert any("returning-0" in set(entities[va]) and "returning-0" not in set(entities[tr])
               for tr, va in splits)


def test_forward_splits_refuse_missing_or_insufficient_times():
    with pytest.raises(ValueError, match="nonmissing"):
        forward_group_splits(["2020-01-01", None], ["a", "b"], min_train_fraction=.5)
    with pytest.raises(ValueError, match="distinct timestamps"):
        forward_group_splits(["2020-01-01"] * 10, np.arange(10), min_train_fraction=.5)


@pytest.mark.parametrize("entities", [["p1", None], ["p1", ""]])
def test_forward_splits_refuse_missing_entity_ids(entities):
    with pytest.raises(ValueError, match="entity_ids"):
        forward_group_splits(
            ["2020-01-01", "2020-01-02"],
            np.asarray(entities, dtype=object),
            min_train_fraction=.5,
        )


def test_grouped_random_cv_is_optimistic_under_known_temporal_drift():
    rng = np.random.default_rng(2301)
    n = 400
    dev_times = np.repeat(pd.date_range("2024-01-01", periods=2, freq="D"), n)
    dev_entities = np.arange(2 * n).astype(str)
    dev_y = np.repeat([0., 5.], n) + rng.normal(0, 1., 2 * n)
    future_times = pd.date_range("2024-01-03", periods=n, freq="min")
    future_y = 20. + rng.normal(0, 1., n)
    X_dev = np.ones((len(dev_y), 1)); X_future = np.ones((n, 1))

    legacy_oof = np.empty(len(dev_y))
    for fit, val in GroupKFold(5).split(X_dev, dev_y, dev_entities):
        legacy_oof[val] = DummyRegressor().fit(X_dev[fit], dev_y[fit]).predict(X_dev[val])
    forward_oof = np.full(len(dev_y), np.nan)
    for fit, val in forward_group_splits(dev_times, dev_entities, n_splits=1,
                                         min_train_fraction=.5):
        forward_oof[val] = DummyRegressor().fit(X_dev[fit], dev_y[fit]).predict(X_dev[val])
    deployed = DummyRegressor().fit(X_dev, dev_y).predict(X_future)
    legacy_mae = np.mean(np.abs(dev_y - legacy_oof))
    forward_mae = np.nanmean(np.abs(dev_y - forward_oof))
    future_mae = np.mean(np.abs(future_y - deployed))
    assert legacy_mae < forward_mae < future_mae
