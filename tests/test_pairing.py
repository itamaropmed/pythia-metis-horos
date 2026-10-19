import numpy as np
import pytest

from horos.represent.pairing import (
    clustered_nearest_pairs, distance_refined_pairs, global_nearest_pairs,
)


def test_global_nearest_pairs_are_deterministic_disjoint_and_complete():
    rng = np.random.default_rng(42)
    x = rng.uniform(size=(100, 4))
    first = global_nearest_pairs(x, np.arange(100))
    second = global_nearest_pairs(x, np.arange(100))
    endpoints = np.concatenate([first.left_indices, first.right_indices])
    assert np.array_equal(np.sort(endpoints), np.arange(100))
    assert np.array_equal(first.left_indices, second.left_indices)
    assert first.meta["design_hash"] == second.meta["design_hash"]
    assert np.all((0 <= first.distances) & (first.distances <= 1))
    assert first.meta["pairing_outcome_blind"] is False


def test_distance_refinement_is_outcome_free_complete_and_nonincreasing():
    x = np.random.default_rng(94).uniform(size=(64, 4))
    original = global_nearest_pairs(x, np.arange(64))
    refined = distance_refined_pairs(x, np.arange(64), max_passes=2)
    assert np.array_equal(
        np.sort(np.r_[refined.left_indices, refined.right_indices]), np.arange(64)
    )
    assert sum(refined.distances) <= sum(original.distances) + 1e-10
    assert refined.meta["initial_mean_pair_distance"] == pytest.approx(
        np.mean(original.distances)
    )


def test_global_nearest_pairs_refuses_odd_or_duplicate_entities():
    with pytest.raises(ValueError, match="even"):
        global_nearest_pairs(np.zeros((5, 2)), np.arange(5))
    with pytest.raises(ValueError, match="one row"):
        global_nearest_pairs(np.zeros((4, 2)), np.array([0, 1, 1, 3]))


def test_global_nearest_pairs_handles_repeated_feature_values_without_self_pairs():
    x = np.repeat(np.linspace(0, 1, 20), 4)[:, None]
    design = global_nearest_pairs(x, np.arange(len(x)))
    endpoints = np.concatenate([design.left_indices, design.right_indices])
    assert np.array_equal(np.sort(endpoints), np.arange(len(x)))
    assert np.all(design.left_indices != design.right_indices)
    assert np.all(design.distances == 0)


def test_clustered_nearest_pairs_never_cross_clusters_and_drop_outcome_free():
    rng = np.random.default_rng(43)
    x = rng.uniform(size=(15, 3))
    clusters = np.repeat(np.array(["a", "b", "c"]), 5)
    design = clustered_nearest_pairs(x, np.arange(15), clusters)
    assert design.meta["n_dropped_entities"] == 3
    assert design.meta["n_used_entities"] == 12
    assert np.all(clusters[design.left_indices] == clusters[design.right_indices])
    assert np.array_equal(clusters[design.left_indices], design.pair_blocks)
