import numpy as np
import pytest

from pythia import finite_refinement_permutation_test


def call(coarse, refinement, y, **kwargs):
    return finite_refinement_permutation_test(
        coarse, refinement, y, loss="mae", permutations=199, seed=71,
        registry_frozen_before_outcomes=True, outcome_blind_attested=True,
        conditional_exchangeability_attested=True,
        provenance={"registry": "synthetic-v1"}, **kwargs)


def test_informative_refinement_is_detected_without_training():
    rng = np.random.default_rng(5)
    coarse = np.repeat(np.arange(8), 80)
    refinement = rng.integers(0, 2, len(coarse))
    y = coarse + 2. * refinement + rng.uniform(-.2, .2, len(coarse))
    result = call(coarse, refinement, y)
    assert result.status == "complete"
    assert result.observed_gain > result.null_quantiles["q99"]
    assert result.p_value == pytest.approx(1 / 200)


def test_independent_refinement_does_not_create_large_signal_in_fixed_seed():
    rng = np.random.default_rng(17)
    coarse = np.repeat(np.arange(10), 100)
    refinement = rng.integers(0, 4, len(coarse))
    y = coarse + rng.normal(0, 1, len(coarse))
    result = call(coarse, refinement, y)
    assert result.p_value > .05


def test_bijective_relabeling_is_numerically_invariant():
    rng = np.random.default_rng(8)
    coarse = np.repeat(np.arange(5), 60)
    refinement = rng.integers(0, 3, len(coarse))
    y = coarse + refinement + rng.normal(0, .5, len(coarse))
    first = call(coarse, refinement, y)
    encoded = np.asarray([f"cipher-{37 * int(value) + 9}" for value in refinement])
    second = call(coarse, encoded, y)
    assert first.observed_gain == pytest.approx(second.observed_gain)
    assert first.p_value == second.p_value
    assert first.null_quantiles == second.null_quantiles


def test_target_derived_or_unfrozen_refinement_refuses():
    y = np.linspace(0, 1, 30)
    result = finite_refinement_permutation_test(
        np.zeros(30), y.astype(str), y, loss="mae", permutations=19, seed=1,
        registry_frozen_before_outcomes=True, outcome_blind_attested=False,
        conditional_exchangeability_attested=True,
        provenance={"registry": "literal-target"})
    assert result.status == "refused"
    assert "outcome-blind" in result.refusal["reason"]


def test_pinball_and_window_losses_are_supported():
    rng = np.random.default_rng(9)
    coarse = np.repeat(np.arange(4), 30)
    refinement = rng.integers(0, 2, len(coarse))
    y = coarse + rng.normal(size=len(coarse))
    common = dict(
        coarse_labels=coarse, refinement_labels=refinement, outcomes=y,
        permutations=19, seed=3, registry_frozen_before_outcomes=True,
        outcome_blind_attested=True, conditional_exchangeability_attested=True,
        provenance={"registry": "multi-loss-v1"})
    pinball = finite_refinement_permutation_test(
        loss="pinball", quantile=.9, **common)
    window = finite_refinement_permutation_test(
        loss="window_miss", window=.5, **common)
    assert 0 < pinball.p_value <= 1
    assert 0 < window.p_value <= 1


def test_unattested_exchangeability_is_labeled_diagnostic_only():
    rng = np.random.default_rng(19)
    coarse = np.repeat(np.arange(3), 20)
    refinement = rng.integers(0, 2, len(coarse))
    y = rng.normal(size=len(coarse))
    result = finite_refinement_permutation_test(
        coarse, refinement, y, loss="mae", permutations=19, seed=2,
        registry_frozen_before_outcomes=True, outcome_blind_attested=True,
        conditional_exchangeability_attested=False,
        exploratory_exchangeability_override=True,
        provenance={"registry": "retrospective-diagnostic"})
    assert result.status == "complete"
    assert result.claim_tier == "exploratory"
    assert "not a valid p-value" in result.assumptions[0]
