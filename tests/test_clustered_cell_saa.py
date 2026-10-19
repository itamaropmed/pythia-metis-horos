import numpy as np
import pytest

from pythia import (
    clustered_finite_cell_saa_lower_bound,
    clustered_fixed_policy_information_value_lower_bound,
    finite_cell_saa_lower_bound,
)


def test_cluster_bound_accounts_for_exact_repeats():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(80), 8)
    cells_by_cluster = rng.integers(0, 8, 80)
    labels = np.repeat(cells_by_cluster, 8)
    shock = rng.uniform(-1, 1, 80)
    y = np.repeat(cells_by_cluster + shock, 8)
    result = clustered_finite_cell_saa_lower_bound(
        labels, y, clusters, loss="mae", outcome_range_bound=None,
        cell_outcome_range_bounds={str(i): 2. for i in range(8)},
        cell_support_provenance="analytic-cell-support-v1", alpha=.05,
        representation_provenance="cell-v1", support_provenance="support-v1",
        independent_clusters_attested=True, outcome_blind_attested=True)
    assert result.status == "complete"
    assert result.effective_cluster_count == pytest.approx(80)
    assert result.maximum_cluster_size == 8
    assert result.bound <= .5 + 1e-12


def test_naive_row_width_is_smaller_than_cluster_valid_width():
    rng = np.random.default_rng(4)
    clusters = np.repeat(np.arange(60), 10)
    labels = np.repeat(rng.integers(0, 6, 60), 10)
    y = labels + np.repeat(rng.uniform(-1, 1, 60), 10)
    common = dict(
        cell_labels=labels, outcomes=y, loss="mae", outcome_range_bound=None,
        cell_outcome_range_bounds={str(i): 2. for i in range(6)},
        cell_support_provenance="support", alpha=.05,
        representation_provenance="cells", support_provenance="support",
        outcome_blind_attested=True)
    clustered = clustered_finite_cell_saa_lower_bound(
        clusters=clusters, independent_clusters_attested=True, **common)
    naive = finite_cell_saa_lower_bound(
        entities=np.arange(len(y)), iid_rows_attested=True, **common)
    assert clustered.concentration_width > naive.concentration_width


def test_clustered_fixed_policy_value_is_positive_and_valid():
    rng = np.random.default_rng(8)
    clusters = np.repeat(np.arange(300), 4)
    concept_by_cluster = rng.integers(0, 2, 300)
    concept = np.repeat(concept_by_cluster, 4)
    shock = np.repeat(rng.uniform(-.5, .5, 300), 4)
    y = 4 * concept + shock
    result = clustered_fixed_policy_information_value_lower_bound(
        np.zeros(len(y)), y, clusters, np.abs(shock), loss="mae",
        loss_range_bound=.5, outcome_range_bound=5., alpha=.05,
        representation_provenance="coarse", support_provenance="[-.5,4.5]",
        policy_provenance="concept-policy", independent_clusters_attested=True,
        outcome_blind_coarse_attested=True,
        policy_frozen_before_evaluation=True,
        policy_refinement_measurable_attested=True,
        policy_preoutcome_attested=True)
    assert result.value_lower_bound > 1
    assert result.value_lower_bound <= 1.75 + 1e-12


def test_missing_cluster_independence_refuses_or_downgrades():
    labels = np.repeat(np.arange(2), 10)
    clusters = np.repeat(np.arange(10), 2)
    y = np.linspace(0, 1, 20)
    common = dict(
        cell_labels=labels, outcomes=y, clusters=clusters, loss="mae",
        outcome_range_bound=1., alpha=.05, representation_provenance="cells",
        support_provenance="support", independent_clusters_attested=False,
        outcome_blind_attested=True)
    refused = clustered_finite_cell_saa_lower_bound(**common)
    exploratory = clustered_finite_cell_saa_lower_bound(
        exploratory_cluster_override=True, **common)
    assert refused.status == "refused"
    assert exploratory.claim_tier == "exploratory"
