import numpy as np

from pythia import PairMatrix
from pythia.core.replicates import (
    paired_replicate_clipped_envelope,
    paired_replicate_ensemble_sensitivity,
    paired_replicate_lower_bound,
    paired_replicate_sensitivity_curve,
)


def _pairs(seed=1, n=10000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    d = rng.uniform(0, .02, n)
    y_left = 2 * x + rng.uniform(-.5, .5, n)
    y_right = 2 * (x + d) + rng.uniform(-.5, .5, n)
    return y_left, y_right, d, np.arange(n), np.arange(n, 2 * n), np.arange(n)


def test_paired_replicate_mae_bound_is_valid_and_nonzero():
    result = paired_replicate_lower_bound(
        *_pairs(), loss="mae", lipschitz_bound=2.0,
        distance_bound=.02, outcome_difference_bound=1.04,
    )
    assert result["status"] == "complete"
    assert 0 < result["bound"] <= .25


def test_paired_replicate_mse_bound_is_valid_and_nonzero():
    result = paired_replicate_lower_bound(
        *_pairs(seed=2), loss="mse", lipschitz_bound=2.0,
        distance_bound=.02, outcome_difference_bound=1.04,
    )
    assert result["status"] == "complete"
    assert 0 < result["bound"] <= 1 / 12


def test_replicate_bound_weakens_with_larger_lipschitz_cap():
    args = _pairs(seed=3)
    values = [
        paired_replicate_lower_bound(
            *args, loss="mae", lipschitz_bound=value,
            distance_bound=.02, outcome_difference_bound=1.1,
        )["bound"]
        for value in (2.0, 3.0, 4.0)
    ]
    assert np.all(np.diff(values) <= 1e-12)


def test_replicate_estimate_is_clipped_to_feasible_nonnegative_risk():
    result = paired_replicate_lower_bound(
        *_pairs(seed=41, n=100), loss="mae", lipschitz_bound=1000.0,
        distance_bound=.02, outcome_difference_bound=1.1,
        concentration="conditional_hoeffding",
    )
    assert result["raw_estimate"] < 0
    assert result["bound"] == result["estimate"] == 0


def test_predeclared_outcome_term_clip_envelope_is_selection_safe_and_sharper():
    args = _pairs(seed=42, n=10000)
    kwargs = dict(
        loss="mae", lipschitz_bound=2.0,
        distance_bound=.02, outcome_difference_bound=20.0,
        concentration="conditional_empirical_bernstein",
    )
    baseline = paired_replicate_lower_bound(*args, **kwargs)
    envelope = paired_replicate_clipped_envelope(
        *args, **kwargs, outcome_term_clips=(.25, .55, 10.0),
    )
    assert envelope["clip_selection_simultaneous"] is True
    assert envelope["confidence"] == .95
    assert envelope["selected_clip"] == .55
    assert baseline["bound"] < envelope["bound"] <= .25
    assert envelope["bound"] == max(row["bound"] for row in envelope["clip_rows"])


def test_conditional_hoeffding_uses_pairs_and_outcome_range_only():
    args = _pairs(seed=31, n=1000)
    result = paired_replicate_lower_bound(
        *args, loss="mae", lipschitz_bound=2.0,
        distance_bound=.02, outcome_difference_bound=1.1,
        concentration="conditional_hoeffding",
    )
    expected_radius = .55 * np.sqrt(np.log(20) / (2 * 1000))
    assert result["status"] == "complete"
    assert result["n_concentration_units"] == 1000
    assert np.isclose(result["radius"], expected_radius)
    assert "audited pair endpoints" in result["estimand"]
    assert all("iid" not in assumption for assumption in result["assumptions"])


def test_conditional_hoeffding_does_not_charge_smoothness_to_sampling_radius():
    args = _pairs(seed=32, n=1000)
    radii = [
        paired_replicate_lower_bound(
            *args, loss="mse", lipschitz_bound=value,
            distance_bound=.02, outcome_difference_bound=1.1,
            concentration="conditional_hoeffding",
        )["radius"]
        for value in (0.0, 2.0, 20.0)
    ]
    assert np.allclose(radii, radii[0])


def test_conditional_empirical_bernstein_is_variance_adaptive_over_pairs():
    args = _pairs(seed=33, n=10000)
    eb = paired_replicate_lower_bound(
        *args, loss="mae", lipschitz_bound=2.0, distance_bound=.02,
        outcome_difference_bound=1.1,
        concentration="conditional_empirical_bernstein",
    )
    hf = paired_replicate_lower_bound(
        *args, loss="mae", lipschitz_bound=2.0, distance_bound=.02,
        outcome_difference_bound=1.1, concentration="conditional_hoeffding",
    )
    assert eb["radius"] < hf["radius"]
    assert eb["n_concentration_units"] == 10000
    assert "Maurer and Pontil" in eb["concentration_reference"]


def test_conditional_methods_do_not_require_randomized_blocks():
    yl, yr, d, left, right, blocks = _pairs(seed=34, n=1000)
    conditional = paired_replicate_lower_bound(
        yl, yr, d, left, right, np.zeros_like(blocks), loss="mae",
        lipschitz_bound=2.0, distance_bound=.02,
        outcome_difference_bound=1.1,
        concentration="conditional_empirical_bernstein",
    )
    assert conditional["status"] == "complete"
    assert conditional["n_independent_blocks"] == 1
    with np.testing.assert_raises_regex(ValueError, "block-level"):
        paired_replicate_lower_bound(
            yl, yr, d, left, right, np.zeros_like(blocks), loss="mae",
            lipschitz_bound=2.0, distance_bound=.02,
            outcome_difference_bound=1.1, concentration="empirical_bernstein",
        )


def test_conditional_curve_uses_one_shared_event_across_lipschitz_grid():
    args = _pairs(seed=35, n=1000)
    single = paired_replicate_lower_bound(
        *args, loss="mae", lipschitz_bound=2.0, distance_bound=.02,
        outcome_difference_bound=1.1,
        concentration="conditional_empirical_bernstein",
    )
    curve = paired_replicate_sensitivity_curve(
        *args, loss="mae", lipschitz_bounds=np.array([2.0, 3.0, 4.0]),
        distance_bound=.02, outcome_difference_bound=1.1,
        concentration="conditional_empirical_bernstein",
    )
    assert np.isclose(curve["curve"][0]["radius"], single["radius"])
    assert "shared outcome-term event" in curve["alpha_allocation"]


def test_conditional_block_method_uses_blocks_and_states_block_estimand():
    yl, yr, d, left, right, _ = _pairs(seed=36, n=1000)
    blocks = np.repeat(np.arange(100), 10)
    result = paired_replicate_lower_bound(
        yl, yr, d, left, right, blocks, loss="mae", lipschitz_bound=2.0,
        distance_bound=.02, outcome_difference_bound=1.1,
        concentration="conditional_block_empirical_bernstein",
    )
    assert result["status"] == "complete"
    assert result["n_concentration_units"] == 100
    assert result["equal_block_sizes"] is True
    assert "pair endpoints" in result["estimand"]
    assert "within blocks is unrestricted" in " ".join(result["assumptions"])


def test_conditional_block_method_discloses_unequal_block_weighting():
    yl, yr, d, left, right, _ = _pairs(seed=37, n=1000)
    blocks = np.concatenate([np.zeros(100, int), np.arange(1, 901)])
    result = paired_replicate_lower_bound(
        yl, yr, d, left, right, blocks, loss="mae", lipschitz_bound=2.0,
        distance_bound=.02, outcome_difference_bound=1.1,
        concentration="conditional_block_empirical_bernstein",
    )
    assert result["equal_block_sizes"] is False
    assert result["pairs_per_block_min"] == 1
    assert result["pairs_per_block_max"] == 100
    assert "equal weighting" in result["estimand"]


def test_replicate_bound_refuses_overlapping_entities_and_broken_caps():
    yl, yr, d, left, right, blocks = _pairs(seed=4, n=10)
    overlap = paired_replicate_lower_bound(
        yl, yr, d, left, left, blocks, loss="mae", lipschitz_bound=2,
        distance_bound=.02, outcome_difference_bound=2,
    )
    broken = paired_replicate_lower_bound(
        yl, yr, d, left, right, blocks, loss="mae", lipschitz_bound=2,
        distance_bound=.001, outcome_difference_bound=2,
    )
    assert overlap["status"] == "refused"
    assert broken["status"] == "refused"


def test_replicate_bound_refuses_unsupported_loss():
    result = paired_replicate_lower_bound(
        *_pairs(seed=5, n=10), loss="auc", lipschitz_bound=2,
        distance_bound=.02, outcome_difference_bound=2,
    )
    assert result["status"] == "refused"


def test_replicate_sensitivity_curve_is_simultaneous_and_monotone():
    curve = paired_replicate_sensitivity_curve(
        *_pairs(seed=6), loss="mae", lipschitz_bounds=np.array([2.0, 3.0, 4.0]),
        distance_bound=.02, outcome_difference_bound=1.1,
    )
    values = np.asarray([row["bound"] for row in curve["curve"]])
    assert curve["simultaneous"] is True
    assert curve["confidence"] == .95
    assert np.all(np.diff(values) <= 1e-12)


def _pair_design(seed: int, pair_seed: int = 0, n: int = 2000) -> PairMatrix:
    rng = np.random.default_rng(seed)
    entity = np.arange(n)
    x = rng.uniform(size=n)
    y = 2 * x + rng.uniform(-.5, .5, n)
    order = np.random.default_rng(pair_seed).permutation(n).reshape(-1, 2)
    left, right = order[:, 0], order[:, 1]
    return PairMatrix(
        y[left], y[right], np.abs(x[left] - x[right]), entity[left], entity[right],
        np.arange(n // 2), meta={"design_hash": f"seed-{seed}-pair-{pair_seed}"},
    )


def test_replicate_ensemble_selection_is_simultaneous_and_population_aligned():
    designs = (_pair_design(10, 1), _pair_design(10, 2))
    result = paired_replicate_ensemble_sensitivity(
        designs, loss="mae", lipschitz_bounds=np.array([2.0, 3.0]),
        distance_bound=1.0, outcome_difference_bound=3.0,
    )
    assert result["status"] == "complete"
    assert result["selection_safe"] is True
    assert result["n_designs"] == 2
    assert all(row["selected_design_hash"].startswith("seed-10") for row in result["curve"])


def test_replicate_ensemble_refuses_different_populations():
    first = _pair_design(11, 1)
    second = _pair_design(12, 2)
    result = paired_replicate_ensemble_sensitivity(
        (first, second), loss="mae", lipschitz_bounds=np.array([2.0]),
        distance_bound=1.0, outcome_difference_bound=3.0,
    )
    assert result["status"] == "refused"
    assert "identical" in result["reason"]
