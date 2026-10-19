import numpy as np
import pytest

from pythia import fixed_policy_information_value_lower_bound


def call(coarse, y, losses, **kwargs):
    return fixed_policy_information_value_lower_bound(
        coarse, y, np.arange(len(y)), losses, loss="mae",
        loss_range_bound=6., outcome_range_bound=6., alpha=.05,
        representation_provenance="coarse-v1", support_provenance="[-1,5]",
        policy_provenance="frozen-refined-oracle-v1",
        iid_rows_attested=True, outcome_blind_coarse_attested=True,
        policy_frozen_before_evaluation=True,
        policy_refinement_measurable_attested=True,
        policy_preoutcome_attested=True, **kwargs)


def test_fixed_refined_policy_certifies_positive_information_value():
    rng = np.random.default_rng(4)
    n = 3000
    concept = rng.integers(0, 2, n)
    y = 4 * concept + rng.uniform(-.5, .5, n)
    losses = np.abs(y - 4 * concept)
    result = call(np.zeros(n), y, losses)
    assert result.status == "complete"
    assert result.value_lower_bound > 1
    assert result.value_lower_bound <= 1.75 + 1e-12
    assert result.diagnostics["model_training_performed"] is False
    assert result.diagnostics["alpha_spent"] == pytest.approx(.05)


def test_no_information_policy_returns_zero_not_negative_certificate():
    rng = np.random.default_rng(9)
    y = rng.uniform(-1, 1, 1000)
    result = call(np.zeros(len(y)), y, np.abs(y))
    assert result.value_lower_bound == 0
    assert result.raw_value_endpoint < 0


def test_outcome_derived_or_nonmeasurable_policy_refuses():
    y = np.linspace(-1, 1, 100)
    result = fixed_policy_information_value_lower_bound(
        np.zeros(len(y)), y, np.arange(len(y)), np.zeros(len(y)), loss="mae",
        loss_range_bound=2., outcome_range_bound=2., alpha=.05,
        representation_provenance="coarse", support_provenance="support",
        policy_provenance="literal-target", iid_rows_attested=True,
        outcome_blind_coarse_attested=True,
        policy_frozen_before_evaluation=False,
        policy_refinement_measurable_attested=True,
        policy_preoutcome_attested=False)
    assert result.status == "refused"
    assert "policy frozen" in result.refusal["reason"]


def test_all_registered_upper_routes_are_simultaneous_and_selected():
    rng = np.random.default_rng(12)
    concept = rng.integers(0, 2, 1500)
    y = 4 * concept + rng.uniform(-.5, .5, len(concept))
    result = call(np.zeros(len(y)), y, np.abs(y - 4 * concept))
    assert {row["method"] for row in result.upper_candidates} == {
        "Hoeffding", "empirical_Bernstein", "Bentkus", "dKelly"}
    assert result.policy_risk_upper == min(
        row["upper"] for row in result.upper_candidates)


def test_unattested_sampling_is_explicitly_exploratory():
    rng = np.random.default_rng(15)
    concept = rng.integers(0, 2, 500)
    y = 4 * concept + rng.uniform(-.5, .5, len(concept))
    result = fixed_policy_information_value_lower_bound(
        np.zeros(len(y)), y, np.arange(len(y)), np.abs(y - 4 * concept),
        loss="mae", loss_range_bound=6., outcome_range_bound=6., alpha=.05,
        representation_provenance="coarse", support_provenance="support",
        policy_provenance="policy", iid_rows_attested=False,
        outcome_blind_coarse_attested=True,
        policy_frozen_before_evaluation=True,
        policy_refinement_measurable_attested=True,
        policy_preoutcome_attested=True, exploratory_sampling_override=True)
    assert result.status == "complete"
    assert result.claim_tier == "exploratory"
