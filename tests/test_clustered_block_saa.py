import numpy as np

from pythia import clustered_block_saa_portfolio


def _common(labels, y, groups):
    return dict(
        cell_labels=labels, outcomes=y, clusters=groups, loss="mae",
        block_cluster_sizes=(10, 20, 40), score_caps=(.25, .5, 1., 2.),
        alpha=.05, partition_seed=17,
        representation_provenance="frozen-cells-v1",
        partition_provenance="seeded-cluster-permutation-v1",
        independent_clusters_attested=True,
        outcome_blind_labels_attested=True,
        partition_frozen_before_outcomes=True)


def test_support_free_block_saa_is_positive_and_below_uniform_floor():
    rng = np.random.default_rng(10)
    groups = np.arange(800)
    labels = rng.integers(0, 4, len(groups))
    y = 5 * labels + rng.uniform(-1, 1, len(groups))
    result = clustered_block_saa_portfolio(**_common(labels, y, groups))
    assert result.status == "conditional"
    assert 0 < result.bound <= .5
    assert result.diagnostics["population_outcome_support_required"] is False
    assert result.diagnostics["model_training_performed"] is False


def test_extreme_outcome_needs_no_population_range():
    rng = np.random.default_rng(11)
    groups = np.arange(800)
    labels = np.zeros(len(groups), dtype=int)
    y = rng.choice((-1e6, 0., 1e6), len(groups), p=(.1, .8, .1))
    result = clustered_block_saa_portfolio(**_common(labels, y, groups))
    assert result.status == "conditional"
    assert np.isfinite(result.bound)
    assert result.bound >= 0


def test_missing_independence_or_outcome_blindness_refuses():
    groups = np.arange(100)
    labels = np.zeros(100, dtype=int)
    y = np.linspace(-1, 1, 100)
    common = _common(labels, y, groups)
    common["independent_clusters_attested"] = False
    refused = clustered_block_saa_portfolio(**common)
    assert refused.status == "refused"
    assert "independent clusters" in refused.refusal["reason"]
    common["exploratory_cluster_override"] = True
    exploratory = clustered_block_saa_portfolio(**common)
    assert exploratory.status == "exploratory"
    assert exploratory.diagnostics["cluster_independence_attested"] is False


def test_routes_are_deterministic_and_alpha_is_ledgered():
    rng = np.random.default_rng(12)
    groups = np.arange(200)
    labels = rng.integers(0, 3, 200)
    y = labels + rng.uniform(-1, 1, 200)
    first = clustered_block_saa_portfolio(**_common(labels, y, groups))
    second = clustered_block_saa_portfolio(**_common(labels, y, groups))
    assert first.to_dict() == second.to_dict()
    assert first.alpha_per_candidate == .05 / (3 * 4)
