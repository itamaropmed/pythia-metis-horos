import numpy as np
import pytest

from pythia.core.concentration import bentkus_lower_mean
from pythia.core.group_witness import grouped_decision_lower_bound
from pythia.core.replicates import paired_replicate_lower_bound


def test_bentkus_lower_mean_endpoints_and_domain():
    assert bentkus_lower_mean(np.zeros(20), 1., .05) == 0
    all_one = bentkus_lower_mean(np.ones(20), 1., .05)
    assert all_one == pytest.approx((.05 / np.e) ** (1 / 20))
    with pytest.raises(ValueError, match="outside"):
        bentkus_lower_mean(np.array([0., 1.1]), 1., .05)


def test_bentkus_pair_and_group_invariants():
    x = np.zeros((80, 1))
    y = np.tile([0., 100.], 40)
    e = np.arange(80)
    pair = paired_replicate_lower_bound(
        y[::2], y[1::2], np.zeros(40), e[::2], e[1::2],
        np.arange(40), loss="pinball", quantile=.9,
        lipschitz_bound=0., distance_bound=1.,
        outcome_difference_bound=100., concentration="conditional_bentkus",
    )
    group = grouped_decision_lower_bound(
        x, y, e, np.arange(80).reshape(20, 4), loss="pinball",
        quantile=.9, lipschitz_bound=0., outcome_range_bound=100.,
        concentration="conditional_bentkus",
    )
    assert 0 <= pair["bound"] <= pair["estimate"]
    assert 0 <= group["bound"] <= group["estimate"]
    assert pair["bound"] <= 5.
    assert group["bound"] <= 5.


def test_bentkus_lower_mean_nonidentical_bernoulli_coverage():
    rng = np.random.default_rng(20260913)
    probabilities = np.linspace(.05, .45, 80)
    truth = float(np.mean(probabilities))
    values = [bentkus_lower_mean(
        (rng.random(80) < probabilities).astype(float), 1., .05,
    ) for _ in range(1000)]
    assert sum(value <= truth for value in values) >= 945
