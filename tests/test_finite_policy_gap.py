import numpy as np
import pytest

from pythia import (
    finite_coordinate_bernstein_gap_upper_bound,
    finite_policy_gap_resolution,
    finite_policy_gap_upper_bound,
    finite_loss_routed_gap_upper_bound,
)


LOSS01 = np.ones((2, 2)) - np.eye(2)


def result_for(y, z, g, alpha=.05):
    return finite_policy_gap_upper_bound(
        LOSS01[np.asarray(y, int)], z, g, loss="zero_one",
        action_labels=(0, 1), z_alphabet=(0, 1), g_alphabet=(0, 1),
        loss_bounds=(0., 1.), alpha=alpha,
        registry_frozen_before_evaluation=True,
        action_class_complete_for_target=True,
        provenance={"registry": "unit-test"}, maximum_bayes_risk=.5)


def test_policy_gap_independence_and_disclosure_behave_logically():
    g = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 2500)
    z = np.tile([0, 0, 1, 1, 0, 0, 1, 1], 2500)
    independent = result_for(np.tile([0, 1, 0, 1, 0, 1, 0, 1], 2500), z, g)
    disclosed = result_for(z, z, g)
    assert independent.status == "conditional"
    assert independent.plugin_gap == pytest.approx(0., abs=1e-12)
    assert disclosed.plugin_gap == pytest.approx(.5)
    assert disclosed.gap_cap == .5
    assert independent.gap_cap < disclosed.gap_cap


def test_policy_preflight_does_not_pay_for_outcome_alphabet():
    plan = finite_policy_gap_resolution(
        n=3000, g_alphabet_size=2, z_alphabet_size=2,
        action_count=3, loss_range=1., alpha=.05)
    assert plan["outcome_alphabet_required"] is False
    assert plan["alpha_coarse"] + plan["alpha_fine"] == pytest.approx(.05)
    assert 0 < plan["gap_uncertainty"] < .1


def test_policy_route_refuses_posthoc_or_incomplete_action_classes():
    kwargs = dict(
        losses=[[0., 1.], [1., 0.]], z=[0, 1], g=[0, 0],
        loss="zero_one", action_labels=(0, 1), z_alphabet=(0, 1),
        g_alphabet=(0,), loss_bounds=(0., 1.), provenance={"registry": "x"})
    posthoc = finite_policy_gap_upper_bound(
        **kwargs, registry_frozen_before_evaluation=False,
        action_class_complete_for_target=True)
    incomplete = finite_policy_gap_upper_bound(
        **kwargs, registry_frozen_before_evaluation=True,
        action_class_complete_for_target=False)
    assert posthoc.status == incomplete.status == "refused"


def test_policy_gap_covers_random_population_values():
    rng = np.random.default_rng(20261017)
    violations = 0
    for _ in range(500):
        p = rng.dirichlet(np.full(8, .7)).reshape(2, 2, 2)
        flat = rng.choice(8, size=3000, p=p.ravel())
        y, z, g = flat // 4, (flat // 2) % 2, flat % 2
        result = result_for(y, z, g)
        coarse = np.einsum("yzg,ya->ga", p, LOSS01)
        fine = np.einsum("yzg,ya->gza", p, LOSS01)
        truth = float(np.min(coarse, axis=1).sum() - np.min(fine, axis=2).sum())
        violations += result.gap_cap + 1e-12 < truth
    assert violations / 500 < .06


def test_coordinate_bernstein_gap_covers_random_population_values():
    rng = np.random.default_rng(20261018)
    violations = 0
    for _ in range(500):
        p = rng.dirichlet(np.full(8, .7)).reshape(2, 2, 2)
        flat = rng.choice(8, size=3000, p=p.ravel())
        y, z, g = flat // 4, (flat // 2) % 2, flat % 2
        result = finite_coordinate_bernstein_gap_upper_bound(
            LOSS01[y], z, g, loss="zero_one", action_labels=(0, 1),
            z_alphabet=(0, 1), g_alphabet=(0, 1), loss_bounds=(0., 1.),
            alpha=.05, registry_frozen_before_evaluation=True,
            action_class_complete_for_target=True,
            provenance={"registry": "random-eb-check"}, maximum_bayes_risk=.5)
        coarse = np.einsum("yzg,ya->ga", p, LOSS01)
        fine = np.einsum("yzg,ya->gza", p, LOSS01)
        truth = float(np.min(coarse, axis=1).sum() - np.min(fine, axis=2).sum())
        violations += result.gap_cap + 1e-12 < truth
    assert violations / 500 < .06


def test_coordinate_route_records_simultaneous_event_geometry():
    y = np.tile([0, 0, 1, 1], 1000)
    z = np.tile([0, 1, 0, 1], 1000)
    g = np.zeros(len(y), int)
    result = finite_coordinate_bernstein_gap_upper_bound(
        LOSS01[y], z, g, loss="zero_one", action_labels=(0, 1),
        z_alphabet=(0, 1), g_alphabet=(0,), loss_bounds=(0., 1.),
        alpha=.05, registry_frozen_before_evaluation=True,
        action_class_complete_for_target=True,
        provenance={"registry": "geometry"}, maximum_bayes_risk=.5)
    assert result.status == "conditional"
    assert result.diagnostics["coordinate_event_count"] == 6
    assert "Maurer-Pontil" in result.diagnostics["concentration"]


def test_frozen_loss_router_selects_without_current_outcomes():
    residual = np.arange(3)[:, None] - np.arange(3)[None, :]
    y = np.tile([0, 1, 2, 0], 1000)
    z = np.tile([0, 0, 1, 1], 1000)
    g = np.zeros(len(y), int)
    common = dict(
        z=z, g=g, action_labels=(0, 1, 2), z_alphabet=(0, 1),
        g_alphabet=(0,), registry_frozen_before_evaluation=True,
        action_class_complete_for_target=True,
        provenance={"registry": "router"})
    mae_matrix = np.abs(residual)
    mae = finite_loss_routed_gap_upper_bound(
        mae_matrix[y], loss="MAE", loss_bounds=(0., 2.), **common)
    q90_matrix = np.maximum(.9 * residual, -.1 * residual)
    q90 = finite_loss_routed_gap_upper_bound(
        q90_matrix[y], loss="Q90", loss_bounds=(0., 1.8), **common)
    assert mae.diagnostics["selected_route"] == "finite_policy_hoeffding"
    assert q90.diagnostics["selected_route"] == "coordinate_empirical_bernstein"
    assert mae.diagnostics["route_selected_from_current_outcomes"] is False
    with pytest.raises(ValueError, match="not registered"):
        finite_loss_routed_gap_upper_bound(
            mae_matrix[y], loss="Q75", loss_bounds=(0., 2.), **common)
