import numpy as np
import pytest

from pythia import finite_cell_saa_lower_bound


def call(labels, y, *, loss="mae", **kwargs):
    return finite_cell_saa_lower_bound(
        labels, y, np.arange(len(y)), loss=loss,
        outcome_range_bound=None if loss == "window_miss" else 2.0,
        alpha=.05, representation_provenance="frozen-cell-parser-v1",
        support_provenance="uniform-hard-support-v1",
        iid_rows_attested=True, outcome_blind_attested=True, **kwargs)


def test_bijective_relabeling_leaves_finite_cell_bound_exactly_unchanged():
    rng = np.random.default_rng(91)
    labels = np.repeat(np.arange(20), 20)
    y = np.repeat(np.linspace(-.5, .5, 20), 20) + rng.uniform(-.5, .5, 400)
    first = call(labels, y)
    encoded = np.array([f"cipher-{37 * int(v) + 11}" for v in labels])
    second = call(encoded, y)
    assert first.bound == pytest.approx(second.bound)
    assert first.empirical_saa_score == pytest.approx(second.empirical_saa_score)


def test_irrelevant_unique_tokens_preserve_validity_but_destroy_tightness():
    rng = np.random.default_rng(12)
    labels = np.repeat(np.arange(20), 20)
    y = np.repeat(np.linspace(-.5, .5, 20), 20) + rng.uniform(-.5, .5, 400)
    baseline = call(labels, y)
    fragmented = call([f"{cell}-id-{i}" for i, cell in enumerate(labels)], y)
    assert baseline.bound > 0
    assert fragmented.bound == 0
    assert fragmented.singleton_fraction == 1


def test_target_derived_cells_are_refused():
    y = np.linspace(-1, 1, 100)
    result = finite_cell_saa_lower_bound(
        y.astype(str), y, np.arange(100), loss="mae", outcome_range_bound=2,
        representation_provenance="literal-target-text", support_provenance="[-1,1]",
        iid_rows_attested=True, outcome_blind_attested=False)
    assert result.status == "refused"
    assert "outcome-blind" in result.reason


def test_known_uniform_mae_floor_is_not_exceeded_in_seeded_battery():
    # For U[-.5,.5], Bayes MAE is .25. This is a deterministic regression
    # check across 100 fresh datasets, not a replacement for a coverage proof.
    misses = 0
    for seed in range(100):
        rng = np.random.default_rng(5000 + seed)
        labels = np.repeat(np.arange(40), 16)
        locations = np.repeat(np.linspace(-.4, .4, 40), 16)
        result = call(labels, locations + rng.uniform(-.5, .5, len(labels)))
        misses += result.bound > .25 + 1e-12
    assert misses == 0


def test_pinball_and_window_losses_are_supported():
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(12), 12)
    y = rng.uniform(-1, 1, len(labels))
    pinball = call(labels, y, loss="pinball", quantile=.9)
    window = call(labels, y, loss="window_miss", window=.25)
    assert 0 <= pinball.bound <= pinball.empirical_saa_score
    assert 0 <= window.bound <= window.empirical_saa_score <= 1


def test_invalid_support_and_sampling_contract_fail_closed():
    labels = np.repeat(np.arange(2), 2)
    y = np.array([-2., 0., 0., 2.])
    refused = call(labels, y)
    assert refused.status == "refused"
    missing = finite_cell_saa_lower_bound(
        labels, np.zeros(4), np.arange(4), loss="mae", outcome_range_bound=2,
        representation_provenance="cells", support_provenance="support",
        iid_rows_attested=False, outcome_blind_attested=True)
    assert missing.status == "refused"


def test_cell_specific_ranges_recover_high_signal_small_noise():
    rng = np.random.default_rng(77)
    labels = np.repeat(np.arange(20), 20).astype(str)
    locations = np.repeat(np.linspace(-5, 5, 20), 20)
    y = locations + rng.uniform(-.15, .15, len(labels))
    common = dict(
        cell_labels=labels, outcomes=y, entities=np.arange(len(y)), loss="mae",
        alpha=.05, representation_provenance="cells-v1",
        support_provenance="global-support-v1", iid_rows_attested=True,
        outcome_blind_attested=True)
    global_result = finite_cell_saa_lower_bound(
        outcome_range_bound=10.3, **common)
    local_result = finite_cell_saa_lower_bound(
        outcome_range_bound=None,
        cell_outcome_range_bounds={str(i): .3 for i in range(20)},
        cell_support_provenance="analytic-cell-supports-v1", **common)
    assert global_result.bound == 0
    assert local_result.bound > 0
    assert local_result.bound <= .075 + 1e-12
    assert local_result.concentration_width < global_result.concentration_width
    assert local_result.concentration_route == "cell_specific_range"


def test_cell_specific_ranges_require_complete_honest_provenance():
    labels = np.array(["a", "a", "b", "b"])
    y = np.array([0., 1., 0., 2.])
    common = dict(
        cell_labels=labels, outcomes=y, entities=np.arange(4), loss="mae",
        alpha=.05, representation_provenance="cells", support_provenance="global",
        iid_rows_attested=True, outcome_blind_attested=True, outcome_range_bound=None)
    with pytest.raises(ValueError, match="provenance"):
        finite_cell_saa_lower_bound(
            cell_outcome_range_bounds={"a": 1., "b": 2.}, **common)
    with pytest.raises(ValueError, match="cover every"):
        finite_cell_saa_lower_bound(
            cell_outcome_range_bounds={"a": 1.}, cell_support_provenance="cell-v1",
            **common)
    refused = finite_cell_saa_lower_bound(
        cell_outcome_range_bounds={"a": .5, "b": 2.},
        cell_support_provenance="cell-v1", **common)
    assert refused.status == "refused"


def test_unattested_sampling_can_only_produce_explicit_exploratory_diagnostic():
    labels = np.repeat(np.arange(4), 10)
    y = np.tile(np.linspace(-.5, .5, 10), 4)
    result = finite_cell_saa_lower_bound(
        labels, y, np.arange(len(y)), loss="mae", outcome_range_bound=1.,
        alpha=.05, representation_provenance="cells-v1",
        support_provenance="unit-support-v1", iid_rows_attested=False,
        outcome_blind_attested=True, exploratory_sampling_override=True)
    assert result.status == "complete"
    assert result.claim_tier == "exploratory"
    assert "no finite-sample coverage claim" in result.assumptions[0]
