import numpy as np
import pytest

from pythia import (
    mixture_dkelly_pinball_portfolio,
    overlapping_pinball_barycentric_portfolio,
)


def _partitions(n=40, k=10, repeats=4):
    rng = np.random.default_rng(11)
    return np.stack([rng.permutation(n).reshape(-1, k)
                     for _ in range(repeats)])


def test_overlapping_partitions_are_allowed_and_entity_sensitivity_is_exact():
    y = np.linspace(0, 1, 40)
    result = overlapping_pinball_barycentric_portfolio(
        y, {10: _partitions()}, quantile=.9, clip=(0, 1),
        declared_discrepancy_by_k={10: 0}, score_caps=(.02,),
        alpha_family=.05, partition_provenance="balanced-random-v1",
        discrepancy_provenance="exact-law cells",
        portfolio_frozen_before_outcomes=True)
    assert result.status == "conditional"
    candidate = result.candidates[0]
    assert candidate["n_partitions"] == 4
    assert candidate["entity_bounded_difference_scale"] == pytest.approx(.2)
    assert result.to_dict()["bound"] == result.bound


def test_repeating_the_same_frozen_partition_does_not_fake_more_confidence():
    y = np.linspace(0, 1, 40)
    partition = _partitions(repeats=1)
    common = dict(
        outcomes=y, quantile=.9, clip=(0, 1),
        declared_discrepancy_by_k={10: 0}, score_caps=(.1,),
        partition_provenance="registered", discrepancy_provenance="exact",
        portfolio_frozen_before_outcomes=True)
    once = overlapping_pinball_barycentric_portfolio(
        partitions_by_k={10: partition}, **common)
    repeated = overlapping_pinball_barycentric_portfolio(
        partitions_by_k={10: np.repeat(partition, 8, axis=0)}, **common)
    assert repeated.selected_radius == pytest.approx(once.selected_radius)
    assert repeated.selected_center == pytest.approx(once.selected_center)


def test_constant_outcomes_give_zero_and_posthoc_registry_refuses():
    common = dict(
        outcomes=np.ones(40), partitions_by_k={10: _partitions()},
        quantile=.9, clip=(0, 2), declared_discrepancy_by_k={10: 0},
        score_caps=(.1,), alpha_family=.05,
        partition_provenance="registered", discrepancy_provenance="exact")
    result = overlapping_pinball_barycentric_portfolio(
        **common, portfolio_frozen_before_outcomes=True)
    assert result.bound == 0
    refused = overlapping_pinball_barycentric_portfolio(
        **common, portfolio_frozen_before_outcomes=False)
    assert refused.status == "refused"


def test_invalid_partition_or_missing_declaration_fails_closed():
    partitions = _partitions()
    partitions[0, 0, 0] = partitions[0, 0, 1]
    with pytest.raises(ValueError, match="cover each entity"):
        overlapping_pinball_barycentric_portfolio(
            np.linspace(0, 1, 40), {10: partitions}, quantile=.9,
            clip=(0, 1), declared_discrepancy_by_k={10: 0},
            score_caps=(.1,), partition_provenance="registered",
            discrepancy_provenance="exact",
            portfolio_frozen_before_outcomes=True)


def test_mixture_dkelly_does_not_multiply_overlapping_evalues():
    y = np.linspace(0, 1, 40)
    partition = _partitions(repeats=1)
    common = dict(
        outcomes=y, quantile=.9, clip=(0, 1),
        declared_partition_discrepancies={10: 0.}, score_caps=(.08,),
        partition_provenance="registered", discrepancy_provenance="exact",
        portfolio_frozen_before_outcomes=True)
    once = mixture_dkelly_pinball_portfolio(
        partitions_by_k={10: partition}, **common)
    repeated = mixture_dkelly_pinball_portfolio(
        partitions_by_k={10: np.repeat(partition, 12, axis=0)}, **common)
    assert repeated.bound == pytest.approx(once.bound)
    assert repeated.diagnostics["products_across_overlapping_partitions_used"] is False


def test_mixture_dkelly_partition_penalties_are_monotone():
    y = np.linspace(0, 1, 40)
    partitions = _partitions(repeats=4)
    common = dict(
        outcomes=y, partitions_by_k={10: partitions}, quantile=.9,
        clip=(0, 1), score_caps=(.08,), partition_provenance="registered",
        discrepancy_provenance="external", portfolio_frozen_before_outcomes=True)
    zero = mixture_dkelly_pinball_portfolio(
        declared_partition_discrepancies={10: np.zeros(4)}, **common)
    positive = mixture_dkelly_pinball_portfolio(
        declared_partition_discrepancies={10: np.full(4, .02)}, **common)
    assert positive.bound <= zero.bound
    assert positive.selected_penalty == pytest.approx(.01)


def test_weighted_evalue_mixture_requires_provenance_and_one_hot_matches_partition():
    y = np.linspace(0, 1, 40)
    partitions = _partitions(repeats=4)
    common = dict(
        outcomes=y, partitions_by_k={10: partitions}, quantile=.9,
        clip=(0, 1), declared_partition_discrepancies={10: np.zeros(4)},
        score_caps=(.08,), partition_provenance="registered",
        discrepancy_provenance="exact", portfolio_frozen_before_outcomes=True)
    with pytest.raises(ValueError, match="weights and provenance"):
        mixture_dkelly_pinball_portfolio(
            **common, partition_weights_by_k={10: [1, 0, 0, 0]})
    weighted = mixture_dkelly_pinball_portfolio(
        **common, partition_weights_by_k={10: [1, 0, 0, 0]},
        weight_provenance="frozen one-hot geometry weight")
    single = mixture_dkelly_pinball_portfolio(
        **{**common, "partitions_by_k": {10: partitions[:1]},
           "declared_partition_discrepancies": {10: [0.]}})
    assert weighted.bound == pytest.approx(single.bound)
    assert weighted.candidates[0]["effective_partition_count"] == pytest.approx(1.)
