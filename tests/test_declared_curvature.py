import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from pythia import bayes_action_jensen_dispersion, declared_action_curvature_profile


def _normal_pinball_risk(action, mean, scale, q):
    """E rho_q(Y-action) for a normal law, in the package's pinball scale."""
    z = (action - mean) / scale
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return scale * phi + (mean - action) * (q - ndtr(z))


def test_curvature_profile_uses_tighter_valid_penalty():
    result = declared_action_curvature_profile(
        lower_score_mean=8.0,
        retained_fraction=0.75,
        loss="Q90",
        declared_action_variances=[4.0, 9.0],
        curvature_bounds=[0.2, 0.2],
        declared_action_mads=[1.0, 2.0],
        action_dispersion_provenance="external-twin-v1",
        curvature_provenance="registered-density-envelope-v1",
    )
    assert result.claim_tier == "conditional"
    assert [p.quadratic_penalty for p in result.points] == pytest.approx([0.4, 0.9])
    assert [p.linear_penalty for p in result.points] == pytest.approx([0.9, 1.8])
    assert [p.lower_bound for p in result.points] == pytest.approx([5.7, 5.325])


def test_local_curvature_fails_closed_outside_registered_radius():
    common = dict(
        lower_score_mean=2.0, retained_fraction=1.0, loss="MAE",
        declared_action_variances=[0.5], curvature_bounds=[0.4],
        action_dispersion_provenance="declared-v", curvature_provenance="local-density-v",
        curvature_radius=1.0,
    )
    with pytest.raises(ValueError, match="requires maximum"):
        declared_action_curvature_profile(**common)
    with pytest.raises(ValueError, match="leaves the curvature domain"):
        declared_action_curvature_profile(max_action_deviations=[1.1], **common)
    accepted = declared_action_curvature_profile(max_action_deviations=[0.9], **common)
    assert accepted.points[0].curvature_radius == 1.0


def test_mse_curvature_convention_is_enforced():
    common = dict(
        lower_score_mean=5, retained_fraction=1, loss="MSE",
        declared_action_variances=[2],
        action_dispersion_provenance="v", curvature_provenance="analytic",
    )
    with pytest.raises(ValueError, match="curvature 2"):
        declared_action_curvature_profile(curvature_bounds=[1.99], **common)
    result = declared_action_curvature_profile(curvature_bounds=[2], **common)
    assert result.points[0].selected_penalty == pytest.approx(2)
    assert result.points[0].lower_bound == pytest.approx(3)


@pytest.mark.parametrize("q,loss", [(0.5, "MAE"), (0.1, "Q10"), (0.9, "Q90")])
def test_normal_population_jensen_gap_is_below_curvature_penalty(q, loss):
    means = np.array([-0.8, -0.1, 0.4, 1.2])
    scales = np.array([0.8, 1.1, 0.9, 1.3])
    zq = {-1: 0.0, 0.1: -1.2815515655446004, 0.5: 0.0,
          0.9: 1.2815515655446004}[q]
    actions = means + scales * zq
    centre = float(actions.mean())
    variance = float(np.mean((actions - centre) ** 2))
    multiplier = 2.0 if loss == "MAE" else 1.0
    individual = np.array([
        multiplier * _normal_pinball_risk(a, m, s, q)
        for a, m, s in zip(actions, means, scales)
    ]).mean()
    objective = lambda a: np.mean([
        multiplier * _normal_pinball_risk(a, m, s, q)
        for m, s in zip(means, scales)
    ])
    mixture = minimize_scalar(objective, bounds=(-5, 5), method="bounded").fun
    kappa = multiplier * (1 / (np.sqrt(2 * np.pi) * scales.min()))
    penalty = 0.5 * kappa * variance
    assert mixture - individual <= penalty + 1e-10


def test_bad_inputs_and_missing_provenance_are_rejected():
    with pytest.raises(ValueError, match="provenance"):
        declared_action_curvature_profile(
            lower_score_mean=1, retained_fraction=1, loss="MAE",
            declared_action_variances=[1], curvature_bounds=[1],
            action_dispersion_provenance="", curvature_provenance="density")
    with pytest.raises(ValueError, match="aligned"):
        declared_action_curvature_profile(
            lower_score_mean=1, retained_fraction=1, loss="MAE",
            declared_action_variances=[1, 2], curvature_bounds=[1],
            action_dispersion_provenance="v", curvature_provenance="density")


def test_optimized_jensen_geometry_dominates_arithmetic_centre():
    actions = np.array([-2.0, 0.0, 3.0])
    kappas = np.array([0.1, 1.0, 4.0])
    result = bayes_action_jensen_dispersion(
        actions, kappas, action_lipschitz=1.0)
    arithmetic = actions.mean()
    arithmetic_penalty = np.mean(0.5 * kappas * (actions - arithmetic) ** 2)
    assert result.quadratic_penalty < arithmetic_penalty
    assert result.quadratic_center == pytest.approx(np.dot(actions, kappas) / kappas.sum())
    assert result.linear_center == 0.0
    assert result.selected_penalty <= result.linear_penalty


def test_jensen_geometry_rejects_invalid_declarations():
    with pytest.raises(ValueError, match="aligned"):
        bayes_action_jensen_dispersion([0, 1], [1])
    with pytest.raises(ValueError, match="Lipschitz"):
        bayes_action_jensen_dispersion([0, 1], [1, 1], action_lipschitz=-1)
